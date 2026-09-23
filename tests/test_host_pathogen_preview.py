"""The preview uses real measurements, field identities and recorded planes."""

import hashlib
import sqlite3

import numpy as np
import pytest

from spacr.host_pathogen_example import build_example
from spacr.host_pathogen import default_settings
from spacr.host_pathogen_preview import preview_fields, preview_field
from spacr.qt.settings_pack import settings_from_pack


@pytest.fixture
def sample(tmp_path):
    folder = build_example(tmp_path / 'sample')
    config, _ = settings_from_pack('host_pathogen', folder / 'settings', defaults=default_settings())
    return folder, config


def test_sample_preview_has_exact_counts_ratios_masks_and_no_writes(sample):
    folder, config = sample
    before = {str(p): hashlib.sha256(p.read_bytes()).digest() for p in folder.rglob('*') if p.is_file()}
    fields, limited = preview_fields(config)
    assert len(fields) == 4 and not limited
    result = preview_field(config, fields[0])
    assert result['image'].shape == (256, 384)
    assert result['planes'] == {'host': 4, 'vacuole': 6, 'parasite': 7}
    assert np.unique(result['masks']['vacuole']).tolist() == list(range(8))
    vacuoles = result['results']['vacuoles'].set_index('vacuole_id')
    assert vacuoles.parasite_count.tolist() == [1, 2, 4, 8, 0, 2, 4]
    assert vacuoles.loc[5, 'channel_0_state'] == 'unknown'
    assert vacuoles.loc[7, 'channel_1_state'] == 'unknown'
    assert vacuoles.loc[1, 'channel_0_recruitment_ratio'] == .8
    cells = result['results']['cells']
    assert len(cells) == 6 and cells.infected.sum() == 5
    assert len(result['results']['orphan_parasites']) == 1
    after = {str(p): hashlib.sha256(p.read_bytes()).digest() for p in folder.rglob('*') if p.is_file()}
    assert before == after


def test_fields_bound_sources_and_queries_and_reject_duplicate_sources(sample, tmp_path):
    folder, config = sample
    fields, limited = preview_fields(config, limit=2)
    assert len(fields) == 2 and limited
    with pytest.raises(ValueError, match='preview limit'):
        preview_field(config, fields[0], row_limit=5)
    second = build_example(tmp_path / 'second')
    fields, limited = preview_fields(dict(config, src=[str(folder), str(second)]))
    assert len(fields) == 8 and not limited
    assert len({f['database'] for f in fields}) == 2
    with pytest.raises(ValueError, match='same project'):
        preview_fields(dict(config, src=[str(folder), str(folder)]))


def test_current_thresholds_change_calls_but_not_stored_measurements(sample):
    _, config = sample
    fields, _ = preview_fields(config)
    original = preview_field(config, fields[0])
    changed = preview_field(dict(config, hp_marker_thresholds={0: .5, 1: 4}), fields[0])
    a, b = [r['results']['vacuoles'] for r in (original, changed)]
    np.testing.assert_array_equal(a.channel_0_recruitment_ratio, b.channel_0_recruitment_ratio)
    assert a.channel_0_state.iloc[0] == 'negative' and b.channel_0_state.iloc[0] == 'positive'
    assert b.channel_0_state.iloc[4] == 'unknown'


def test_missing_or_invalid_images_do_not_hide_measured_results(sample):
    folder, config = sample
    fields, _ = preview_fields(config)
    result = preview_field(config, fields[0], image_channel=99)
    assert result['image'] is None and 'outside' in result['image_note']
    assert len(result['results']['vacuoles']) == 7
    for p in (folder / 'merged').glob('*.npy'):
        p.unlink()
    result = preview_field(config, fields[0])
    assert result['image'] is None and 'No matching' in result['image_note']
    assert len(result['results']['vacuoles']) == 7


def test_legacy_planes_are_explicit_and_bad_planes_are_explained(sample):
    folder, config = sample
    fields, _ = preview_fields(config)
    (folder / 'merged' / '.spacr_plane_layout.json').unlink()
    result = preview_field(config, fields[0])
    assert result['image'] is not None and result['masks'] == {}
    result = preview_field(config, fields[0], planes={'vacuole': 6, 'host': 4})
    assert set(result['masks']) == {'vacuole', 'host'}
    result = preview_field(config, fields[0], planes={'vacuole': 99})
    assert result['image'] is None and 'mask plane' in result['image_note']


def test_empty_fields_keep_uninfected_hosts_and_table_names_are_quoted(sample):
    folder, config = sample
    database = folder / 'measurements' / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('DELETE FROM pathogen WHERE columnID="c1" AND fieldID="1"')
    fields, _ = preview_fields(config)
    result = preview_field(config, fields[0])
    assert len(result['results']['vacuoles']) == 0
    assert result['results']['wells'].infection_fraction.iloc[0] == 0
    with pytest.raises(ValueError, match='not found'):
        preview_field(dict(config, hp_vacuole_table='pathogen"; DROP TABLE cell;--'), fields[0])
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 24


def test_field_with_vacuoles_but_no_hosts_is_still_available(sample):
    folder, config = sample
    database = folder / 'measurements' / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('DELETE FROM cell WHERE columnID="c1" AND fieldID="1"')
    fields, _ = preview_fields(config)
    assert len(fields) == 4
    result = preview_field(config, fields[0])
    assert len(result['results']['cells']) == 0
    assert len(result['results']['vacuoles']) == 7
    assert result['results']['vacuoles'].channel_0_state.eq('unknown').all()


def test_invalid_limits_missing_sources_and_incomplete_identity_fail_clearly(sample, tmp_path):
    folder, config = sample
    database = folder / 'measurements' / 'measurements.db'
    fields, _ = preview_fields(dict(config, src=str(database)))
    with pytest.raises(ValueError, match='field limit'):
        preview_fields(config, limit=0)
    with pytest.raises(ValueError, match='object limit'):
        preview_field(config, fields[0], row_limit=0)
    with pytest.raises(FileNotFoundError, match='Measurements database'):
        preview_fields(dict(config, src=str(tmp_path / 'missing')))
    with pytest.raises(ValueError, match='nonempty measurement table'):
        preview_fields(dict(config, hp_vacuole_table=''))
    with sqlite3.connect(database) as connection:
        connection.execute('ALTER TABLE pathogen RENAME COLUMN fieldID TO missing_field')
    with pytest.raises(ValueError, match='lacks plate/well/field'):
        preview_fields(config)
    with pytest.raises(ValueError, match='lacks field/time'):
        preview_field(config, fields[0])
    with sqlite3.connect(database) as connection:
        connection.execute('ALTER TABLE cell RENAME COLUMN fieldID TO missing_field')
    with pytest.raises(ValueError, match='Host measurements lack'):
        preview_fields(config)


@pytest.mark.parametrize('damage, explanation', [
    ('flat', 'H × W × planes'),
    ('fractional_mask', 'integer labels'),
    ('mask_as_image', 'intensity channel'),
])
def test_bad_image_content_keeps_analysis_available(sample, damage, explanation):
    _, config = sample
    fields, _ = preview_fields(config)
    original = preview_field(config, fields[0])
    image_channel = 0
    if damage == 'mask_as_image':
        image_channel = 4
    else:
        data = np.load(original['path']).astype(float)
        if damage == 'flat':
            data = data[:, :, 0]
        else:
            data[0, 0, 6] = .5
        np.save(original['path'], data)
    result = preview_field(config, fields[0], image_channel=image_channel)
    assert result['image'] is None
    assert explanation in result['image_note']
    assert len(result['results']['cells']) == 6
    assert len(result['results']['vacuoles']) == 7
