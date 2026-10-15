"""An explicit calibration reference is exact, reproducible and read only."""
import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from spacr import intensity_rescale as rescale

KEY = 'intensity_calibration_reference_plate'


def sessions(root):
    merged = root / 'merged'
    merged.mkdir(parents=True)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[4:12, 4:12] = 1
    for plate, level in [('plate1', 600), ('plate2', 1100)]:
        for well in ('A01', 'B02'):
            np.save(merged / f'{plate}_{well}_1.npy', np.stack([
                np.full_like(labels, level), labels], axis=-1))
    settings = {'src': str(merged), 'channels': [0], 'cell_mask_dim': 1,
                'nucleus_mask_dim': None, 'pathogen_mask_dim': None,
                'organelle_mask_dim': None, 'intensity_calibration': True,
                'intensity_calibration_wells': ['A01'],
                'intensity_calibration_statistic': 'median',
                'intensity_calibration_offset': 100, 'timelapse': False,
                'test_mode': False, 'cell_min_size': 0, 'nucleus_min_size': 0,
                'pathogen_min_size': 0, 'cytoplasm_min_size': 0,
                'save_png': False, 'save_arrays': False, 'plot': False,
                'verbose': False, 'n_jobs': 1}
    return merged, settings


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npy')}


def test_named_reference_controls_gains_labels_and_provenance(tmp_path):
    merged, settings = sessions(tmp_path)
    settings[KEY] = 'plate2'
    original, before = copy.deepcopy(settings), hashes(merged)
    files = sorted(before)
    plan = rescale._build_calibration_plan(merged, files, settings)
    assert plan['reference_plate'] == 'plate2'
    assert plan['plates']['plate1']['gain'] == {'0': 2.0}
    assert plan['plates']['plate2']['gain'] == {'0': 1.0}
    data = np.load(merged / 'plate1_B02_1.npy')
    calibrated, record = rescale._apply_calibration(data, 'plate1_B02_1.npy', {
        **settings, rescale.CALIBRATION_SETTINGS_KEY: plan})
    assert np.all(calibrated[..., 0] == 1100)
    assert np.array_equal(calibrated[..., 1], data[..., 1])
    assert record['reference_plate'] == 'plate2' and record['gain'] == {'0': 2.0}
    assert settings == original and hashes(merged) == before


@pytest.mark.parametrize('selection', ['', None, '  '])
def test_blank_selection_preserves_first_plate_default(tmp_path, selection):
    merged, settings = sessions(tmp_path)
    settings[KEY] = selection
    plan = rescale._build_calibration_plan(merged, hashes(merged), settings)
    assert plan['reference_plate'] == 'plate1'
    assert plan['plates']['plate1']['gain'] == {'0': 1.0}
    assert plan['plates']['plate2']['gain'] == {'0': .5}


@pytest.mark.parametrize('selection', ['missing', 'PLATE2', 'plate1,plate2', ['plate1', 'plate2'], True])
def test_missing_or_ambiguous_selection_refuses_without_writes(tmp_path, selection):
    merged, settings = sessions(tmp_path)
    settings[KEY] = selection
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match=KEY):
        rescale._build_calibration_plan(merged, hashes(merged), settings)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_alpha_registry_and_csv_json_roundtrip(tmp_path):
    from spacr import settings as registry
    from spacr.cli import load_settings_file, write_settings_file

    defaults = registry.get_measure_crop_settings({})
    assert defaults[KEY] == ''
    assert KEY in registry.expected_types and isinstance(None, registry.expected_types[KEY])
    assert KEY in registry.categories['Intensity Calibration α']
    assert KEY in registry.ALPHA_FEATURES[580]['settings']
    assert 'IntensityCalibrationReferencePlate' in registry.ALPHA_FEATURES[580]['widgets']
    settings = {'src': '/dataset', KEY: 'plate2'}
    target = tmp_path / 'selected.csv'
    write_settings_file(settings, target)
    assert load_settings_file(str(target)) == settings
    assert load_settings_file(str(target.with_suffix('.json'))) == settings


def test_actual_measurement_records_reference_and_changed_selection_refuses_resume(tmp_path, monkeypatch):
    from spacr import database_concurrency, io, measure
    from spacr.settings import get_measure_crop_settings

    merged, settings = sessions(tmp_path)
    settings[KEY] = 'plate2'
    resolved = get_measure_crop_settings(copy.deepcopy(settings))
    before = hashes(merged)
    measure.measure_crop(resolved)
    measure.measure_crop(get_measure_crop_settings({**settings, 'resume': True}))
    database = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        rows = connection.execute('SELECT intensity_calibration FROM intensity_rescale').fetchall()
    records = [json.loads(row[0]) for row in rows]
    assert records and {record['reference_plate'] for record in records} == {'plate2'}
    old_bytes = database.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail('Changed calibration reached database/resume mutation')

    monkeypatch.setattr(database_concurrency, 'enable_wal_where_safe', forbidden)
    monkeypatch.setattr(measure, 'plan_measure_resume', forbidden)
    monkeypatch.setattr(io, '_save_settings_to_db', forbidden)
    with pytest.raises(ValueError, match='different or unverified intensity calibration'):
        measure.measure_crop(get_measure_crop_settings({**settings, KEY: 'plate1', 'resume': True}))
    assert database.read_bytes() == old_bytes
    assert hashes(merged) == before
