"""Actual calibration inputs distinguish acquisitions and external references."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile

from spacr.intensity_rescale import (
    CALIBRATION_SETTINGS_KEY, _apply_calibration, _build_calibration_plan,
)


def _field(folder, name, signal):
    data = np.zeros((8, 8, 2), dtype=np.uint16)
    data[..., 0] = 100 + signal
    np.save(folder / name, data)
    return data


def _settings(**changes):
    return {
        'intensity_calibration': True,
        'intensity_calibration_wells': ['A01'],
        'intensity_calibration_statistic': 'median',
        'intensity_calibration_offset': 100,
        'cell_mask_dim': 1,
        **changes,
    }


def test_one_plate_has_two_recorded_sessions_with_separate_gains(tmp_path):
    source = tmp_path / 'merged'
    source.mkdir()
    names = []
    for field, signal in [(1, 1000), (2, 2500)]:
        for well in ['A01', 'B02']:
            name = f'plate1_{well}_{field}.npy'
            _field(source, name, signal)
            names.append(name)
    sessions = tmp_path / 'acquisitions.csv'
    with sessions.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'acquisition'])
        writer.writerows((name, f'day{1 if name.endswith("_1.npy") else 2}')
                         for name in names)
    settings = _settings(intensity_calibration_sessions=str(sessions),
                         intensity_calibration_session_column='acquisition')
    settings[CALIBRATION_SETTINGS_KEY] = _build_calibration_plan(source, names, settings)
    means = []
    for field in [1, 2]:
        name = f'plate1_B02_{field}.npy'
        measured, provenance = _apply_calibration(np.load(source / name), name, settings)
        means.append(float(measured[..., 0].mean()))
        np.testing.assert_array_equal(measured[..., 1], np.zeros((8, 8)))
    assert means == [1100.0, 1100.0]


def test_external_tiff_beads_calibrate_sample_only_plate_folders(tmp_path):
    source = tmp_path / 'merged'
    source.mkdir()
    beads = tmp_path / 'beads'
    beads.mkdir()
    names = []
    for plate, signal in [('plate1', 1000), ('plate2', 2000)]:
        name = f'{plate}_B02_1.npy'
        _field(source, name, signal)
        names.append(name)
        tifffile.imwrite(beads / f'{plate}_A01_1.tif',
                         np.full((8, 8), signal + 100, dtype=np.uint16))
    settings = _settings(intensity_calibration_beads=str(beads))
    settings[CALIBRATION_SETTINGS_KEY] = _build_calibration_plan(source, names, settings)
    for name in names:
        measured, provenance = _apply_calibration(np.load(source / name), name, settings)
        np.testing.assert_array_equal(measured[..., 0], np.full((8, 8), 1100))
        np.testing.assert_array_equal(measured[..., 1], np.zeros((8, 8)))


def _recorded_inputs(tmp_path, *, external=False):
    source = tmp_path / 'merged'
    source.mkdir()
    beads = tmp_path / 'beads'
    beads.mkdir()
    names = []
    rows = []
    for day, signal in [(1, 1000), (2, 2500)]:
        for well in ['A01', 'B02']:
            name = f'plate1_{well}_{day}.npy'
            if external and well == 'A01':
                name = name.removesuffix('.npy') + '.tif'
                tifffile.imwrite(beads / name, np.full((8, 8), signal + 100, dtype=np.uint16))
            else:
                _field(source, name, signal)
                names.append(name)
            rows.append([name, f'day{day}'])
    sessions = tmp_path / 'acquisitions.csv'
    _write_table(sessions, rows)
    return source, names, sessions, beads, rows


def _write_table(path, rows, headers=('filename', 'acquisition')):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _recorded_settings(sessions, beads=None, **changes):
    return _settings(intensity_calibration_sessions=str(sessions),
                     intensity_calibration_session_column='acquisition',
                     intensity_calibration_beads=str(beads) if beads else '', **changes)


def test_explicit_session_and_consistent_plate_select_the_actual_gain_reference(tmp_path):
    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    settings = _recorded_settings(sessions,
        intensity_calibration_reference_plate='plate1',
        intensity_calibration_reference_session='day2')
    plan = _build_calibration_plan(source, names, settings)
    assert plan['reference_session'] == 'day2'
    assert plan['sessions']['day1']['gain'] == {'0': 2.5}
    data = np.load(source / 'plate1_B02_1.npy')
    measured, record = _apply_calibration(data, 'plate1_B02_1.npy', {
        **settings, CALIBRATION_SETTINGS_KEY: plan})
    assert measured[..., 0].mean() == 2600
    assert record['sessionID'] == 'day1'
    assert record['reference_session'] == 'day2'
    assert record['plateID'] == record['reference_plate'] == 'plate1'


def test_a_plate_with_two_sessions_refuses_an_ambiguous_reference_selection(tmp_path):
    import pytest

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    with pytest.raises(ValueError, match='session_selection'):
        _build_calibration_plan(source, names, _recorded_settings(sessions,
            intensity_calibration_reference_plate='plate1'))


def test_an_unknown_session_refuses_before_changing_any_source(tmp_path):
    import pytest

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    before = {path: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(ValueError, match='session_selection'):
        _build_calibration_plan(source, names, _recorded_settings(sessions,
            intensity_calibration_reference_session='absent'))
    assert all(path.read_bytes() == data for path, data in before.items())


def test_session_csv_refuses_duplicate_or_missing_assignments_and_headers(tmp_path):
    import pytest

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    cases = [(rows + [rows[0]], ('filename', 'acquisition'), 'row'),
             (rows[:-1], ('filename', 'acquisition'), 'missing_filename'),
             ([['', 'day1'], *rows[1:]], ('filename', 'acquisition'), 'row'),
             ([[rows[0][0], ''], *rows[1:]], ('filename', 'acquisition'), 'row'),
             (rows, ('filename', 'wrong'), 'headers'),
             (rows, ('filename', 'filename'), 'headers'),
             ([[f'../{rows[0][0]}', 'day1'], *rows[1:]], ('filename', 'acquisition'), 'row')]
    for values, headers, reason in cases:
        _write_table(sessions, values, headers)
        before = sessions.read_bytes()
        with pytest.raises(ValueError, match=reason):
            _build_calibration_plan(source, names, _recorded_settings(sessions))
        assert sessions.read_bytes() == before


def test_external_tiff_channel_axis_maps_to_intensity_planes_only(tmp_path):
    source = tmp_path / 'merged'
    source.mkdir()
    beads = tmp_path / 'beads'
    beads.mkdir()
    names = []
    for day, signal in [(1, 1000), (2, 2000)]:
        data = np.zeros((8, 8, 3), dtype=np.uint16)
        data[..., 0] = signal + 100
        data[..., 1] = signal // 2 + 100
        data[..., 2] = 17
        name = f'plate{day}_B02_1.npy'
        np.save(source / name, data)
        names.append(name)
        reference = np.stack([data[..., 0], data[..., 1]])
        tifffile.imwrite(beads / f'plate{day}_A01_1.tif', reference,
                         metadata={'axes': 'CYX'}, photometric='minisblack')
    settings = _settings(cell_mask_dim=2, intensity_calibration_beads=str(beads))
    settings[CALIBRATION_SETTINGS_KEY] = _build_calibration_plan(source, names, settings)
    data = np.load(source / names[1])
    measured, record = _apply_calibration(data, names[1], settings)
    np.testing.assert_array_equal(measured[..., 0], np.full((8, 8), 1100))
    np.testing.assert_array_equal(measured[..., 1], np.full((8, 8), 600))
    np.testing.assert_array_equal(measured[..., 2], data[..., 2])
    assert record['gain'] == {'0': .5, '1': .5}


def test_external_tiff_ambiguous_time_axis_and_wrong_channel_count_refuse(tmp_path):
    import pytest

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path, external=True)
    path = next(beads.iterdir())
    for axes, shape, reason in [('TYX', (2, 8, 8), 'tiff_axes'),
                                ('CYX', (2, 8, 8), 'channels')]:
        tifffile.imwrite(path, np.full(shape, 1100, dtype=np.uint16),
                         metadata={'axes': axes}, photometric='minisblack')
        with pytest.raises(ValueError, match=reason):
            _build_calibration_plan(source, names, _recorded_settings(sessions, beads))


def test_external_merged_npy_references_keep_the_existing_label_plane_contract(tmp_path):
    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    references = []
    for name in list(names):
        if '_A01_' in name:
            (source / name).rename(beads / name)
            names.remove(name)
            references.append(name)
    settings = _recorded_settings(sessions, beads)
    plan = _build_calibration_plan(source, names, settings)
    assert plan['sessions']['day2']['gain'] == {'0': .4}
    assert all(np.load(beads / name)[..., 1].max() == 0 for name in references)


def test_missing_session_reference_and_mixed_working_units_refuse(tmp_path):
    import pytest

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    missing = 'plate1_A01_2.npy'
    with pytest.raises(ValueError, match='missing_session'):
        _build_calibration_plan(source, [name for name in names if name != missing],
                                _recorded_settings(sessions))
    _write_table(sessions, [[name, 'one'] for name in names])
    data = np.load(source / names[0]).astype(float)
    data[..., 0] = .25
    np.save(source / names[0], data)
    with pytest.raises(ValueError, match='incompatible_working_units'):
        _build_calibration_plan(source, names, _recorded_settings(sessions))


def test_reference_hashes_cover_all_legacy_references_and_external_csv_bytes(tmp_path):
    from spacr.measure import _calibration_reference_hashes

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path, external=True)
    settings = _recorded_settings(sessions, beads, src=str(source))
    before = _calibration_reference_hashes(settings, names)
    assert set(before) == {'sessions_csv', 'beads/plate1_A01_1.tif', 'beads/plate1_A01_2.tif'}
    _write_table(sessions, list(reversed(rows)))
    after = _calibration_reference_hashes(settings, names)
    assert after['sessions_csv'] != before['sessions_csv']
    assert all(after[key] == value for key, value in before.items() if key != 'sessions_csv')
    legacy_source = tmp_path / 'legacy'
    legacy_source.mkdir()
    for name in ['plate1_A01_1.npy', 'plate1_A01_2.npy']:
        _field(legacy_source, name, 1000)
    assert set(_calibration_reference_hashes(_settings(src=str(legacy_source)),
               ['plate1_A01_1.npy', 'plate1_A01_2.npy'])) == {
                   'plate1_A01_1.npy', 'plate1_A01_2.npy'}


def test_real_measurement_sqlite_keeps_session_provenance_and_refuses_changed_reference(tmp_path):
    import json
    import sqlite3
    import pytest
    from spacr import measure
    from spacr.settings import get_measure_crop_settings

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path, external=True)
    for name in names:
        data = np.load(source / name)
        data[2:6, 2:6, 1] = 1
        np.save(source / name, data)
    options = get_measure_crop_settings({})
    options.update(_recorded_settings(sessions, beads, src=str(source), channels=[0],
        nucleus_mask_dim=None, pathogen_mask_dim=None, organelle_mask_dim=None,
        cell_min_size=0, nucleus_min_size=0, pathogen_min_size=0, cytoplasm_min_size=0,
        save_png=False, save_arrays=False, plot=False, verbose=False, n_jobs=1))
    measure.measure_crop(dict(options))
    database = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(database) as connection:
        actual = connection.execute('SELECT cell_channel_0_mean_intensity FROM cell').fetchall()
        records = [json.loads(row[0]) for row in connection.execute(
            'SELECT intensity_calibration FROM intensity_rescale')]
    assert actual == [(1100.0,), (1100.0,)]
    assert {row['sessionID'] for row in records} == {'day1', 'day2'}
    assert {row['reference_session'] for row in records} == {'day1'}
    assert all(set(row['reference_files']) == {'sessions_csv',
        'beads/plate1_A01_1.tif', 'beads/plate1_A01_2.tif'} for row in records)
    measure.measure_crop(dict(options))
    before = database.read_bytes()
    # Item 736 isolates fresh runs; these guards protect continuation instead.
    options['resume'] = True
    _write_table(sessions, list(reversed(rows)))
    with pytest.raises(ValueError, match='unverified intensity'):
        measure.measure_crop(dict(options))
    assert database.read_bytes() == before
    _write_table(sessions, rows)
    reference = beads / 'plate1_A01_2.tif'
    original = reference.read_bytes()
    pixels = tifffile.imread(reference)
    pixels[0, 0] += 1
    tifffile.imwrite(reference, pixels)
    with pytest.raises(ValueError, match='unverified intensity'):
        measure.measure_crop(dict(options))
    assert database.read_bytes() == before
    assert reference.read_bytes() != original


def _input_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('key,value,reason', [
    ('intensity_calibration_sessions', [], 'path'),
    ('intensity_calibration_beads', 12, 'path'),
    ('intensity_calibration_session_column', None, 'column'),
    ('intensity_calibration_session_column', '', 'column'),
    ('intensity_calibration_session_column', '  ', 'column'),
    ('intensity_calibration_session_column', 'filename', 'column'),
    ('intensity_calibration_session_column', 12, 'column'),
    ('intensity_calibration_statistic', 'mode', 'choice'),
    ('intensity_calibration_offset', np.nan, 'finite'),
    ('intensity_calibration_offset', np.inf, 'finite'),
    ('intensity_calibration_offset', -np.inf, 'finite'),
    ('intensity_calibration_reference_plate', ['plate1'], 'selection'),
    ('intensity_calibration_reference_session', 12, 'selection'),
])
def test_invalid_session_options_refuse_without_rewriting_inputs(tmp_path, key, value, reason):
    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    settings = _recorded_settings(sessions, intensity_calibration_reference_session='day1')
    settings[key] = value
    before = _input_bytes(tmp_path)
    with pytest.raises(ValueError, match=reason) as caught:
        _build_calibration_plan(source, names, settings)
    assert key in str(caught.value) or 'reference_session' in str(caught.value)
    assert _input_bytes(tmp_path) == before


@pytest.mark.parametrize('damage,reason', [
    ('empty', 'empty_sources'),
    ('two-dimensional', 'merged_shape'),
    ('labels-only', 'working_units'),
    ('different-session-planes', 'incompatible_planes'),
])
def test_unusable_session_layout_refuses_instead_of_guessing_a_gain(tmp_path, damage, reason):
    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    settings = _recorded_settings(sessions)
    if damage == 'empty':
        names = []
    elif damage == 'two-dimensional':
        np.save(source / names[0], np.zeros((8, 8), np.uint16))
    elif damage == 'labels-only':
        settings['nucleus_mask_dim'] = 0
    else:
        for name in names:
            if name.endswith('_2.npy'):
                data = np.load(source / name)
                np.save(source / name, np.concatenate([data, np.zeros_like(data[..., :1])], axis=-1))
    before = _input_bytes(tmp_path)
    with pytest.raises(ValueError, match=reason):
        _build_calibration_plan(source, names, settings)
    assert _input_bytes(tmp_path) == before


@pytest.mark.parametrize('damage,reason', [
    ('two-dimensional-npy', 'merged_shape'),
    ('wrong-npy-planes', 'planes'),
    ('multiple-tiff-series', 'tiff_series'),
    ('normalized-beads-for-raw-sample', 'working_units'),
    ('reference-overflows-working-range', 'working_units'),
])
def test_external_reference_layout_and_units_are_verified_before_gains(tmp_path, damage, reason):
    source = tmp_path / 'merged'
    source.mkdir()
    beads = tmp_path / 'beads'
    beads.mkdir()
    name = 'plate1_B02_1.npy'
    _field(source, name, 1000)
    if damage == 'two-dimensional-npy':
        np.save(beads / 'plate1_A01_1.npy', np.full((8, 8), 1100, np.uint16))
    elif damage == 'wrong-npy-planes':
        np.save(beads / 'plate1_A01_1.npy', np.full((8, 8, 3), 1100, np.uint16))
    elif damage == 'multiple-tiff-series':
        with tifffile.TiffWriter(beads / 'plate1_A01_1.tif') as writer:
            writer.write(np.full((8, 8), 1100, np.uint16), photometric='minisblack')
            writer.write(np.full((9, 7), 1100, np.uint16), photometric='minisblack')
    else:
        value = .5 if damage == 'normalized-beads-for-raw-sample' else 70000
        tifffile.imwrite(beads / 'plate1_A01_1.tif', np.full((8, 8), value),
                         photometric='minisblack')
    before = _input_bytes(tmp_path)
    with pytest.raises(ValueError, match=reason):
        _build_calibration_plan(source, [name], _settings(intensity_calibration_beads=str(beads)))
    assert _input_bytes(tmp_path) == before


@pytest.mark.parametrize('reference_value', [100.0, 50.0, np.nan])
def test_session_reference_requires_positive_finite_signal_above_offset(tmp_path, reference_value):
    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    path = source / 'plate1_A01_2.npy'
    data = np.load(path).astype(np.float64)
    data[..., 0] = reference_value
    np.save(path, data)
    before = _input_bytes(tmp_path)
    with pytest.raises(ValueError, match='positive_reference'):
        _build_calibration_plan(source, names, _recorded_settings(sessions))
    assert _input_bytes(tmp_path) == before


def test_version_two_plan_roundtrip_is_order_independent_and_binds_each_field(tmp_path):
    import copy
    import json

    source, names, sessions, beads, rows = _recorded_inputs(tmp_path)
    for name in names:
        if '_B02_' in name:
            data = np.load(source / name)
            data[..., 1] = 17
            np.save(source / name, data)
    settings = _recorded_settings(sessions, intensity_calibration_reference_session='day2')
    original_settings = copy.deepcopy(settings)
    before = _input_bytes(tmp_path)
    plan = _build_calibration_plan(source, names, settings)
    assert _input_bytes(tmp_path) == before and settings == original_settings
    _write_table(sessions, list(reversed(rows)))
    reordered_inputs = _input_bytes(tmp_path)
    repeated = _build_calibration_plan(source, list(reversed(names)) + names, settings)
    assert _input_bytes(tmp_path) == reordered_inputs
    assert repeated == plan
    assert plan['version'] == 2 and plan['reference_session'] == 'day2'
    loaded = json.loads(json.dumps(plan, sort_keys=True))
    assert loaded == plan
    for field, expected_gain in [(1, 2.5), (2, 1.0)]:
        name = f'plate1_B02_{field}.npy'
        data = np.load(source / name)
        original = data.copy()
        data.flags.writeable = False
        calibrated, record = _apply_calibration(data, name, {
            **settings, CALIBRATION_SETTINGS_KEY: loaded})
        np.testing.assert_array_equal(data, original)
        assert not np.shares_memory(calibrated, data)
        np.testing.assert_array_equal(calibrated[..., 0], np.full((8, 8), 2600))
        np.testing.assert_array_equal(calibrated[..., 1], original[..., 1])
        assert record['plateID'] == 'plate1' and record['sessionID'] == f'day{field}'
        assert record['reference_session'] == 'day2' and record['gain'] == {'0': expected_gain}
    for unknown in ('plate1_B02_3.npy', 'plate9_B02_1.npy'):
        with pytest.raises(ValueError, match='no calibration gain'):
            _apply_calibration(data, unknown, {**settings, CALIBRATION_SETTINGS_KEY: loaded})
        np.testing.assert_array_equal(data, original)
    assert _input_bytes(tmp_path) == reordered_inputs
    assert settings == original_settings and loaded == plan
