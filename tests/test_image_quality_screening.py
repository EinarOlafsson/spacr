"""Raw-image screening has explicit thresholds, policies and field identities."""
import json

import numpy as np
import pytest

from spacr.image_quality import assess_image, excluded_fields, filter_batch, screen_fields


def fields(tmp_path):
    folder = tmp_path / 'stack'
    folder.mkdir()
    sharp = (np.indices((16, 16)).sum(axis=0) % 2 * 100).astype(np.uint16)[..., None]
    blurred = np.full_like(sharp, 50)
    for name, data in (('sharp.npy', sharp), ('blurred.npy', blurred)):
        np.save(folder / name, data)
    return sharp, blurred


def test_saturation_uses_acquisition_level_not_observed_maximum():
    image = np.zeros((10, 10, 2), np.uint16)
    image[0, 0, 0] = 4000
    image[0, :, 1] = 4095
    records = assess_image(image, dict(image_qc_saturation_level={0: 4095, 1: 4095},
                                       image_qc_max_saturation={0: .01, 1: .01}))
    assert records[0]['saturation_fraction'] == 0
    assert records[0]['reasons'] == []
    assert records[1]['saturation_fraction'] == .1
    assert records[1]['reasons'] == ['saturation_above_threshold']


def test_volume_uses_best_focus_plane_and_nonfinite_is_separate():
    sharp = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float32)
    image = np.stack([np.zeros_like(sharp), sharp])[..., None]
    record = assess_image(image, dict(image_qc_min_focus={0: 1}))[0]
    assert record['focus_variance'] > 1
    assert record['reasons'] == []
    image[0, 1, 1, 0] = np.nan
    assert 'nonfinite_pixels' in assess_image(image, {})[0]['reasons']
    with pytest.raises(ValueError, match='floating images need'):
        assess_image(image, dict(image_qc_max_saturation={0: .1}))


@pytest.mark.parametrize('mode,expected', [('report', []), ('exclude', ['blurred.npy'])])
def test_saved_policy_decides_exclusion_without_changing_pixels(tmp_path, mode, expected):
    sharp, blurred = fields(tmp_path)
    before = {path.name: path.read_bytes() for path in (tmp_path / 'stack').glob('*.npy')}
    policy = dict(image_qc_mode=mode, image_qc_min_focus={0: 1.})
    assert screen_fields(tmp_path, policy) == expected
    assert excluded_fields(tmp_path) == set(expected)
    report = json.loads((tmp_path / 'qc/image_quality.json').read_text())
    assert report['policy']['image_qc_min_focus'] == {'0': 1.}
    assert {row['field'] for row in report['fields']} == set(before)
    bad = next(row for row in report['fields'] if row['field'] == 'blurred.npy')
    assert bad['status'] == ('flagged' if mode == 'report' else 'excluded_image_quality')
    assert all((tmp_path / 'stack' / name).read_bytes() == data for name, data in before.items())
    batch, names = filter_batch(np.stack([blurred, sharp]), ['blurred.npy', 'sharp.npy'],
                                {'image_qc_excluded_fields': expected})
    assert names == (['sharp.npy'] if expected else ['blurred.npy', 'sharp.npy'])
    np.testing.assert_array_equal(batch[-1], sharp)
    assert 'focus_below_threshold' in (tmp_path / 'qc/image_quality.csv').read_text()
    gallery = (tmp_path / 'qc/image_quality.html').read_text()
    assert 'data:image/png;base64,' in gallery
    assert gallery.index('<b>blurred.npy') < gallery.index('<b>sharp.npy')
    assert screen_fields(tmp_path, {'image_qc_mode': 'off'}) == []
    assert excluded_fields(tmp_path) == set()


def test_channel_mapping_is_explicit_and_invalid_policy_fails():
    image = np.zeros((8, 8, 2), np.uint8)
    records = assess_image(image, dict(image_qc_channels=[3], image_qc_min_focus={3: 1}), [1, 3])
    assert len(records) == 1 and records[0]['channel'] == 3
    assert records[0]['saturation_level'] == 255
    with pytest.raises(ValueError, match='not being screened'):
        assess_image(image, dict(image_qc_min_focus={7: 10}))
    with pytest.raises(ValueError, match='absent'):
        assess_image(image, dict(image_qc_channels=[3]))


def test_qc_dashboard_reads_excluded_fields_as_quality_flags(tmp_path):
    from spacr.qt.widgets.qc_summary import read_dashboard

    fields(tmp_path)
    screen_fields(tmp_path, dict(image_qc_mode='exclude', image_qc_min_focus={0: 1}))
    dashboard = read_dashboard(str(tmp_path))
    card = next(card for card in dashboard.cards if card.key == 'image_quality')
    assert card.verdict == 'warn'
    assert '1 excluded' in card.headline
    assert 'not zero-object' in card.detail[0]


def test_mask_settings_expose_a_separate_quality_category():
    from spacr.settings import categories, set_default_settings_preprocess_generate_masks
    from spacr.qt.screens.settings_model import categories_for_app
    from spacr.image_quality import DEFAULTS

    settings = set_default_settings_preprocess_generate_masks({})
    assert settings['image_qc_mode'] == 'off'
    shown = categories_for_app('mask', categories)['Image Quality']
    assert set(DEFAULTS) == set(shown)


def test_exclusion_cannot_silently_leave_old_measured_fields(tmp_path):
    import sqlite3
    fields(tmp_path)
    screen_fields(tmp_path, {'image_qc_mode': 'report'})
    original = (tmp_path / 'qc/image_quality.json').read_bytes()
    (tmp_path / 'measurements').mkdir()
    with sqlite3.connect(tmp_path / 'measurements/measurements.db') as connection:
        connection.execute('CREATE TABLE cell (file_name TEXT, area REAL)')
        connection.execute('INSERT INTO cell VALUES (?, ?)', ('blurred', 42))
    with pytest.raises(ValueError, match='overlap existing measurements'):
        screen_fields(tmp_path, dict(image_qc_mode='exclude', image_qc_min_focus={0: 1}))
    assert (tmp_path / 'qc/image_quality.json').read_bytes() == original
    with sqlite3.connect(tmp_path / 'measurements/measurements.db') as connection:
        assert connection.execute('SELECT * FROM cell').fetchall() == [('blurred', 42)]


def test_v2_filters_before_model_inference(tmp_path, monkeypatch):
    import tifffile
    from spacr import pipeline_v2

    for field, image in [(1, np.zeros((8, 8), np.uint16)),
                         (2, (np.indices((8, 8)).sum(axis=0) % 2 * 100).astype(np.uint16))]:
        tifffile.imwrite(tmp_path / f'plate1_A01_T01F0{field}L01A01Z01C00.tif', image)
    seen = []
    monkeypatch.setattr(pipeline_v2, 'stream_masks_from_stack',
                        lambda stacks, **kwargs: seen.extend(stack.path.name for stack in stacks))
    result = pipeline_v2.run_v2(tmp_path, channels=[0], postprocess_settings=dict(
        image_qc_mode='exclude', image_qc_min_focus={0: 1}))
    rejected = excluded_fields(tmp_path)
    assert len(seen) == 1 and len(rejected) == 1
    assert not set(seen) & rejected
    assert len(result['stacks']) == 1
