"""Item 934: every OPS measurement feature is a Measure feature, from one implementation.

The OPS phenotype path (:func:`spacr.ops_sample.measure_phenotype_field`) and
Measure (:func:`spacr.measure.measure_crop`) compute Funk et al. 2022's mass
displacement, channel-pair Pearson's r and form factor with the same
functions of :mod:`spacr.measure`; Measure gains the OPS switches
(``funk_object_features``, ``zernike``, ``channel_names``,
``measure_gpu_check``, ``measure_gpu_check_rtol``), all under
``ALPHA_FEATURES[934]``, and its default output does not change.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M
from spacr import ops_sample
from spacr.ops_settings import OPS_DEFAULTS, OPS_TO_MEASURE_KEYS

NEW_KEYS = ('funk_object_features', 'zernike', 'channel_names',
            'measure_gpu_check', 'measure_gpu_check_rtol')


def _objects():
    """A uniform object, an off-centre bright spot, a zero-variance channel,
    a one-pixel object and two touching objects."""
    labels = np.zeros((48, 48), np.int64)
    labels[2:12, 2:12] = 1
    labels[2:12, 14:26] = 2
    labels[20:30, 2:14] = 3
    labels[40, 40] = 4
    labels[34:44, 14:22] = 5
    labels[34:44, 22:30] = 6
    rng = np.random.default_rng(934)
    image = rng.random((48, 48, 3)) * 100
    image[2:12, 2:12, 0] = 50.0
    image[3, 22, :] += 5000.0
    image[20:30, 2:14, 2] = 7.0
    return labels, image


def _funk_reference(labels, image):
    """Funk et al.'s definitions object by object: centroid distance, np.corrcoef."""
    rows = []
    for label in np.unique(labels[labels > 0]):
        r, c = np.nonzero(labels == label)
        row = {'label': label}
        for ch in range(image.shape[-1]):
            w = image[r, c, ch]
            row[f'md_{ch}'] = np.hypot((w * r).sum() / w.sum() - r.mean(),
                                       (w * c).sum() / w.sum() - c.mean())
        for a in range(image.shape[-1]):
            for b in range(a + 1, image.shape[-1]):
                with np.errstate(invalid='ignore', divide='ignore'):
                    row[f'r_{a}_{b}'] = (np.corrcoef(image[r, c, a], image[r, c, b])[0, 1]
                                         if r.size > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def test_mass_displacement_and_pearson_equal_funks_definitions():
    labels, image = _objects()
    reference = _funk_reference(labels, image)
    shift = M.measure_mass_displacement(labels, image)
    pairs = M.measure_channel_pair_pearson(labels, image)
    assert list(shift['label']) == list(reference['label'])
    for ch in range(3):
        np.testing.assert_allclose(shift[f'channel_{ch}_mass_displacement'],
                                   reference[f'md_{ch}'], rtol=1e-12, atol=1e-12)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        got = pairs[f'correlation_channel_{a}_channel_{b}'].to_numpy()
        want = reference[f'r_{a}_{b}'].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)],
                                   rtol=1e-12, atol=1e-12)
    flat = pairs.set_index('label')
    assert np.isnan(flat.loc[3, 'correlation_channel_0_channel_2'])
    assert np.isnan(flat.loc[4]).all()


def test_intensity_moments_reads_the_one_mass_displacement():
    labels, image = _objects()
    moments = M.measure_intensity_moments(labels, image)
    shift = M.measure_mass_displacement(labels, image)
    for column in shift.columns:
        assert np.array_equal(moments[column].to_numpy(), shift[column].to_numpy(),
                              equal_nan=True)


def test_channel_pair_pearson_names_and_refuses_a_wrong_count():
    labels, image = _objects()
    named = M.measure_channel_pair_pearson(labels, image, ['DAPI', 'GFP', 'A594'])
    assert list(named.columns) == ['label', 'correlation_dapi_gfp',
                                   'correlation_dapi_a594', 'correlation_gfp_a594']
    with pytest.raises(ValueError, match='2 channel names for 3 channels'):
        M.measure_channel_pair_pearson(labels, image, ['a', 'b'])


def test_ops_funk_features_are_the_measure_functions_renamed():
    labels, image = _objects()
    funk = ops_sample.funk_object_features(labels, image, ['DAPI', 'GFP', 'A594'],
                                           'cell')
    shift = M.measure_mass_displacement(labels, image)
    pairs = M.measure_channel_pair_pearson(labels, image, ['dapi', 'gfp', 'a594'])
    for ch, name in enumerate(('dapi', 'gfp', 'a594')):
        assert np.array_equal(funk[f'cell_{name}_mass_displacement'],
                              shift[f'channel_{ch}_mass_displacement'], equal_nan=True)
    for column in pairs.columns[1:]:
        assert np.array_equal(funk[f'cell_{column}'], pairs[column], equal_nan=True)


def test_form_factor_is_nan_without_a_perimeter():
    values = M._form_factor([100.0, 1.0], [40.0, 0.0])
    assert values[0] == pytest.approx(4 * np.pi * 100 / 1600)
    assert np.isnan(values[1])


def test_channel_names_never_match_inside_a_longer_index():
    names = [f'c{i}' for i in range(12)]
    assert M._channel_named('cell_channel_1_mean', names) == 'cell_c1_mean'
    assert M._channel_named('cell_channel_10_mean', names) == 'cell_c10_mean'
    assert (M._channel_named('cell_channel_0_channel_11_pearson', names)
            == 'cell_c0_c11_pearson')
    assert M._channel_named('cell_channel_12_mean', names) == 'cell_channel_12_mean'


@pytest.mark.parametrize('value, message', [
    (['a', 'b'], '2 names for 3 channels'),
    (['a', 'b b', 'c'], 'cannot name a column'),
    (['a', 'B', 'b'], 'repeats a name'),
])
def test_channel_names_are_checked(value, message):
    with pytest.raises(ValueError, match=message):
        M._measurement_channel_names({'channel_names': value, 'channels': [0, 1, 2]})


def test_channel_names_accept_a_comma_separated_string_and_empty():
    settings = {'channels': [0, 1, 2]}
    assert M._measurement_channel_names(
        {**settings, 'channel_names': 'DAPI, GFP,A594'}) == ['dapi', 'gfp', 'a594']
    for empty in (None, '', []):
        assert M._measurement_channel_names({**settings, 'channel_names': empty}) is None


def _merged_field(size=64):
    yy, xx = np.mgrid[:size, :size]
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros((size, size), np.uint16)
    for i, (cy, cx) in enumerate([(18, 18), (18, 46), (46, 30)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 100] = i
        nucleus[(yy - cy - 2) ** 2 + (xx - cx) ** 2 <= 16] = i
    rng = np.random.default_rng(1)
    chans = []
    for c in range(3):
        base = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        base[cell > 0] += np.uint16(1000 * (c + 1))
        base[nucleus > 0] += rng.integers(0, 900, size=int((nucleus > 0).sum())).astype(np.uint16)
        chans.append(base)
    return np.stack(chans + [cell, nucleus], axis=-1).astype(np.uint16)


def _measure_settings(merged, **over):
    from spacr.settings import get_measure_crop_settings
    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged), 'channels': [0, 1, 2],
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0,
        'cell_mask_dim': 3, 'nucleus_mask_dim': 4, 'pathogen_mask_dim': None,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False, 'test_mode': False,
        'cytoplasm': True, 'n_jobs': 1, 'experiment': 'exp',
    })
    settings.update(over)
    return settings


def _measured(tmp_path, name, **over):
    root = tmp_path / name
    merged = root / 'merged'
    merged.mkdir(parents=True)
    (root / 'measurements').mkdir()
    np.save(merged / 'plate1_A01_F001.npy', _merged_field())
    result = M._measure_crop_core(0, [], 'plate1_A01_F001.npy',
                                  _measure_settings(merged, **over))
    assert result[4] in ('', None), result[4]
    with sqlite3.connect(root / 'measurements' / 'measurements.db') as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {table: pd.read_sql(f'SELECT * FROM "{table}" ORDER BY rowid', conn)
                for table in tables}


def _object_tables(tables):
    return {name: frame for name, frame in tables.items()
            if name in ('cell', 'nucleus', 'cytoplasm')}


def test_default_measure_output_is_unchanged_and_touches_no_new_code(
        tmp_path, monkeypatch):
    """The five new keys at their defaults give the tables of a settings dict
    without them, and none of the new code paths runs."""
    def _refuse(*args, **kwargs):
        raise AssertionError('a 934 code path ran under the default settings')

    monkeypatch.setattr(M, '_add_funk_frames', _refuse)
    monkeypatch.setattr(M, '_rename_measurement_channels', _refuse)
    monkeypatch.setattr(M, 'check_gpu_measurement', _refuse)
    default = _measured(tmp_path, 'default')
    from spacr.settings import get_measure_crop_settings
    defaults = get_measure_crop_settings(settings={})
    assert {key: defaults[key] for key in NEW_KEYS} == {
        'funk_object_features': False, 'zernike': None, 'channel_names': None,
        'measure_gpu_check': True, 'measure_gpu_check_rtol': 1e-4}

    original = M._measure_crop_core

    def _without_new_keys(index, time_ls, file, settings, *args, **kwargs):
        settings = {k: v for k, v in settings.items() if k not in NEW_KEYS}
        return original(index, time_ls, file, settings, *args, **kwargs)

    monkeypatch.setattr(M, '_measure_crop_core', _without_new_keys)
    legacy = _measured(tmp_path, 'legacy')
    default, legacy = _object_tables(default), _object_tables(legacy)
    assert default.keys() == legacy.keys() == {'cell', 'nucleus', 'cytoplasm'}
    for name in default:
        a = default[name].drop(columns=['path_name'])
        b = legacy[name].drop(columns=['path_name'])
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert not [c for c in a.columns
                    if 'mass_displacement' in c or '_correlation_' in c
                    or c.endswith('form_factor')]


def test_measure_writes_the_funk_features_with_channel_names(tmp_path):
    names = ['dapi', 'gfp', 'a594']
    tables = _object_tables(_measured(
        tmp_path, 'funk', funk_object_features=True, channel_names=names))
    plain = _object_tables(_measured(tmp_path, 'plain'))
    field = _merged_field()
    image = field[..., :3].astype(np.float64)
    cell = field[..., 3].astype(np.int64)
    nucleus = field[..., 4].astype(np.int64)
    masks = {'cell': cell, 'nucleus': nucleus,
             'cytoplasm': np.where(nucleus > 0, 0, cell)}
    for role, frame in tables.items():
        frame = frame.set_index('object_label').sort_index()
        assert not [c for c in frame.columns if 'channel_' in c]
        shift = M.measure_mass_displacement(masks[role], image).set_index('label')
        pairs = M.measure_channel_pair_pearson(masks[role], image, names).set_index('label')
        for ch, name in enumerate(names):
            np.testing.assert_array_equal(
                frame[f'{role}_{name}_mass_displacement'].to_numpy(),
                shift.loc[frame.index, f'channel_{ch}_mass_displacement'].to_numpy())
        for column in pairs.columns:
            np.testing.assert_array_equal(frame[f'{role}_{column}'].to_numpy(),
                                          pairs.loc[frame.index, column].to_numpy())
        np.testing.assert_array_equal(
            frame[f'{role}_form_factor'].to_numpy(),
            M._form_factor(frame[f'{role}_area'], frame[f'{role}_perimeter']))
        before = plain[role].set_index('object_label').sort_index()
        for column in before.columns:
            renamed = M._channel_named(column, names)
            if column == 'path_name' or column.startswith('label_list'):
                continue
            pd.testing.assert_series_equal(frame[renamed], before[column],
                                           check_names=False, check_exact=True)


def _intensity_tables(settings, cell, nucleus, image):
    blank = np.zeros_like(cell)
    cytoplasm = np.where(nucleus > 0, 0, cell).astype(np.int32)
    return M._intensity_measurements(cell, nucleus, blank, blank, cytoplasm, image,
                                     settings, sizes=[1, 2, 3, 4, 5],
                                     periphery=False, outside=False)


def test_profile_families_are_not_repeated_by_funk_features():
    field = _merged_field()
    image = field[..., :3].astype(np.float32)
    cell = field[..., 3].astype(np.int32)
    nucleus = field[..., 4].astype(np.int32)
    settings = ops_sample._measurement_settings(
        3, False, profile_features=['intensity_moments', 'shape_extended'])
    tables = _intensity_tables({**settings, 'funk_object_features': True},
                               cell, nucleus, image)
    for frame in tables:
        if not len(frame.columns):
            continue
        features = frame.columns[frame.columns != 'label']
        assert not features.duplicated().any()
        assert sum('mass_displacement' in c for c in features) == 3
        assert sum(c.endswith('_form_factor') for c in features) == 1


def _ops_against_measure(nuclei, cells, image, names, **ops):
    """Every OPS column that Measure writes on the same field, and the ones it does not.

    Measure is run on the field with the settings OPS_TO_MEASURE_KEYS maps the
    OPS switches to; its columns are named as the OPS path names them.
    """
    table = ops_sample.measure_phenotype_field(
        nuclei, cells, image, names, texture=ops.get('texture', True),
        funk_features=ops.get('funk', False), zernike=ops.get('zernike', False),
        profile_features=ops.get('profile_features'))
    mapped = {OPS_TO_MEASURE_KEYS[key]: value for key, value in {
        'ops_ph_texture': ops.get('texture', True),
        'ops_ph_funk_features': ops.get('funk', False),
        'ops_ph_zernike': ops.get('zernike', False),
        'ops_ph_profile_features': ops.get('profile_features'),
    }.items()}
    settings = {**ops_sample._measurement_settings(
        image.shape[2], mapped['homogeneity'],
        profile_features=mapped['profile_features']), **mapped}
    blank = np.zeros_like(cells)
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    morphology = M._morphological_measurements(
        cells, nuclei, blank, blank, cytoplasm, settings,
        zernike=settings['zernike'])
    intensity = M._intensity_measurements(
        cells, nuclei, blank, blank, cytoplasm, image, settings,
        sizes=[1, 2, 3, 4, 5], periphery=False, outside=False)
    lower = [n.lower() for n in names]
    measured = {}
    for slot, prefix in ((0, 'cell'), (1, 'nucleus'), (4, 'cytoplasm')):
        for frame in (morphology[slot], intensity[slot]):
            frame = frame.loc[:, ~frame.columns.duplicated()]
            if 'label' not in frame:
                continue
            frame = frame.set_index(frame['label'].astype(np.int64))
            for column in frame.columns:
                if column == 'label':
                    continue
                name = M._channel_named(str(column), lower)
                if not name.startswith(f'{prefix}_'):
                    name = f'{prefix}_{name}'
                measured.setdefault(name, frame[column])
    missing, unequal = [], []
    for column in table.columns:
        if column in ('label', 'centroid_row', 'centroid_col'):
            continue
        if column not in measured:
            missing.append(column)
            continue
        got = measured[column].reindex(table['label'].to_numpy()).to_numpy(np.float64)
        want = table[column].to_numpy(np.float64)
        if not np.array_equal(got, want, equal_nan=True):
            unequal.append(column)
    return table, missing, unequal


def _ops_field():
    field = _merged_field()
    return (field[..., 4].astype(np.int32), field[..., 3].astype(np.int32),
            field[..., :3].astype(np.float32), ['DAPI', 'GFP', 'A594'])


def test_every_ops_measurement_is_a_measure_feature_with_the_same_value():
    """The ratchet: every column the OPS phenotype path writes, with its
    measurement switches on, is written by Measure with the mapped settings,
    value for value."""
    nuclei, cells, image, names = _ops_field()
    table, missing, unequal = _ops_against_measure(
        nuclei, cells, image, names, texture=True, funk=True,
        profile_features=['edge_intensity', 'intensity_moments'])
    assert len(table.columns) > 150
    assert {'cell_form_factor', 'nucleus_dapi_mass_displacement',
            'cell_correlation_gfp_a594'} <= set(table.columns)
    assert missing == []
    assert unequal == []


def test_the_ratchet_catches_an_ops_only_column(monkeypatch):
    nuclei, cells, image, names = _ops_field()
    original = ops_sample.measure_phenotype_field

    def _planted(*args, **kwargs):
        table = original(*args, **kwargs)
        table['cell_ops_only_feature'] = 1.0
        return table

    monkeypatch.setattr(ops_sample, 'measure_phenotype_field', _planted)
    _table, missing, _unequal = _ops_against_measure(nuclei, cells, image, names,
                                                     texture=False, funk=True)
    assert missing == ['cell_ops_only_feature']


def test_every_ops_measurement_setting_maps_to_a_measure_setting():
    from spacr.settings import ALPHA_FEATURES, get_measure_crop_settings
    measure = get_measure_crop_settings(settings={})
    assert set(OPS_TO_MEASURE_KEYS) == {
        'ops_ph_texture', 'ops_ph_funk_features', 'ops_ph_zernike',
        'ops_ph_measure_gpu', 'ops_ph_measure_gpu_check', 'ops_ph_channels',
        'ops_ph_profile_features', 'ops_ph_granularity_subsample',
        'ops_ph_profile_optional_families'}
    assert set(OPS_TO_MEASURE_KEYS) <= set(OPS_DEFAULTS)
    assert set(OPS_TO_MEASURE_KEYS.values()) <= set(measure)
    assert set(ALPHA_FEATURES[934]['settings']) == set(NEW_KEYS)
    assert not {'profile_features', 'profile_granularity_subsample'} & {
        key for entry in ALPHA_FEATURES.values()
        for key in entry.get('settings', ())}


def test_zernike_true_without_mahotas_stops(monkeypatch):
    def _missing():
        raise ImportError('mahotas is not installed.')

    monkeypatch.setattr(M, '_load_zernike_moments', _missing)
    monkeypatch.setattr(M, '_measurement_device', lambda settings: None)
    cell = np.zeros((32, 32), np.int32)
    cell[4:20, 4:20] = 1
    with pytest.raises(ImportError, match='mahotas'):
        M._morphological_measurements(cell, np.zeros_like(cell), np.zeros_like(cell),
                                      np.zeros_like(cell), np.zeros_like(cell),
                                      {'cell_mask_dim': 0, 'nucleus_mask_dim': None,
                                       'pathogen_mask_dim': None, 'cytoplasm': False},
                                      zernike=True)


def test_zernike_none_and_false_follow_the_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(M, '_ZERNIKE_AVAILABLE', False)
    none = _object_tables(_measured(tmp_path, 'none', zernike=None))
    off = _object_tables(_measured(tmp_path, 'off', zernike=False))
    for name in none:
        assert not [c for c in none[name].columns if 'zernike' in c]
        pd.testing.assert_frame_equal(none[name].drop(columns='path_name'),
                                      off[name].drop(columns='path_name'))


def _fake_measure(perturb):
    def measure(field, settings):
        table = pd.DataFrame({'site': 1, 'label': [1, 2, 3],
                              'a': [1.0, 2.0, 3.0], 'b': [4.0, np.nan, 6.0]})
        if settings['measure_gpu'] and perturb:
            table['b'] = table['b'] * 1.01
        return table
    return measure


def test_gpu_check_names_a_perturbed_column_and_passes_otherwise():
    outside = M.check_gpu_measurement('field', {}, measure=_fake_measure(True))
    assert list(outside['column']) == ['b']
    assert outside['rows_outside'].tolist() == [2]
    assert outside.attrs['summary']['rtol'] == 1e-4
    clean = M.check_gpu_measurement('field', {'measure_gpu_check_rtol': 0.05},
                                    measure=_fake_measure(True))
    assert clean.empty and clean.attrs['summary']['rtol'] == 0.05
    assert M.check_gpu_measurement('field', {}, measure=_fake_measure(False)).empty


def test_gpu_check_reports_an_unreadable_field():
    out = M.check_gpu_measurement('field', {}, measure=lambda f, s: None)
    assert out.empty and out.attrs['summary'] == {'skipped': 'the field is unreadable'}


def test_ops_gpu_check_report_is_unchanged(monkeypatch):
    from spacr import ops_engine
    from spacr.ops_run_infra import compare_tables

    calls = []

    def table(task):
        calls.append(task['measure_gpu'])
        return _fake_measure(True)(None, task), [], (8, 8)

    monkeypatch.setattr(ops_engine, '_ph_measure_table', table)
    report = ops_engine._ph_gpu_check({'site': 7, 'measure_gpu': True})
    assert calls == [True, False]
    gpu, cpu = _fake_measure(True)(None, {'measure_gpu': True}), \
        _fake_measure(True)(None, {'measure_gpu': False})
    assert report == {'site': 7, **compare_tables(gpu, cpu, rtol=1e-4)}
    monkeypatch.setattr(ops_engine, '_ph_measure_table', lambda task: (None, [], None))
    assert ops_engine._ph_gpu_check({'site': 7}) == {
        'site': 7, 'skipped': 'the field is unreadable'}


def test_measure_stops_before_a_gpu_run_that_disagrees(tmp_path, monkeypatch):
    merged = tmp_path / 'merged'
    merged.mkdir()
    np.save(merged / 'plate1_A01_F001.npy', _merged_field())
    settings = _measure_settings(merged, measure_gpu=True)
    monkeypatch.setattr(M, '_measurement_device', lambda settings: 'cuda:0')
    seen = []

    def _tables(field, settings):
        seen.append((field, settings['measure_gpu']))
        return {'cell': _fake_measure(True)(field, settings)}

    monkeypatch.setattr(M, '_field_tables_for_check', _tables)
    with pytest.raises(RuntimeError, match=r'differ beyond rtol 0.0001 in 1 column'):
        M._gpu_check_before_run(settings, ['plate1_A01_F001.npy'])
    assert [s for _f, s in seen] == [True, False]
    assert seen[0][0].endswith('plate1_A01_F001.npy')
    seen.clear()
    M._gpu_check_before_run({**settings, 'measure_gpu_check': False},
                            ['plate1_A01_F001.npy'])
    M._gpu_check_before_run({**settings, 'measure_gpu': False},
                            ['plate1_A01_F001.npy'])
    assert seen == []


def test_the_default_check_measures_a_merged_field_on_both_paths(tmp_path):
    merged = tmp_path / 'plate1_A01_F001.npy'
    np.save(merged, _merged_field())
    settings = _measure_settings(tmp_path)
    outside = M.check_gpu_measurement(str(merged), settings)
    assert outside.empty
    summary = outside.attrs['summary']
    assert set(summary) == {'cell', 'nucleus', 'cytoplasm'}
    assert all(record['rows'] == 3 and record['columns'] > 50
               for record in summary.values())
