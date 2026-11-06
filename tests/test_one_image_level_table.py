"""One image-level table named ``image`` in measurements.db.

Every module that produces a per-field result writes it to the field's row of
``image``, keyed by ``file_name`` (the merged array name without ``.npy``) and
the canonical plate, row, column and field keys:

* Measure: per-channel background outside every object mask and cheap QC
  values, per-mask counts and area fractions, the intensity rescaling and,
  when enabled, confluency;
* the mask stage: its object counts;
* image screening: the rule metrics and classifier verdicts.

The older tables (``confluency``, ``intensity_rescale``, ``object_counts``,
the qc/ CSV) are still written, so their readers keep working. The synthetic
fields have known background, object and saturated pixels.
"""
from __future__ import annotations

import contextlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.measure import (_image_field_values, _image_percentiles,
                           _image_table_background, _image_table_keys,
                           _read_image_table, _write_image_rows)

BACKGROUND, CELL, NUCLEUS = 100, 1000, 2000


def _table(db, name):
    with contextlib.closing(sqlite3.connect(db, timeout=30.0)) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)


def _tables(db):
    with contextlib.closing(sqlite3.connect(db, timeout=30.0)) as conn:
        return {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}


def _field(shift=0, size=64):
    """Two channels, a cell mask and a separate nucleus mask.

    Background is 30% of the field at 100. A cell covers 40% at 1000 and a
    nucleus outside it 30% at 2000, so only excluding both masks recovers 100.
    Channel 1 holds one row of saturated pixels inside the cell.
    """
    cells = np.zeros((size, size), dtype=np.uint16)
    nuclei = np.zeros((size, size), dtype=np.uint16)
    rows = np.arange(size)
    cell_rows = rows < int(0.4 * size) + 1
    nucleus_rows = (rows >= int(0.4 * size) + 1) & (rows < int(0.7 * size) + 1)
    cells[cell_rows, :] = 1
    cells[cell_rows, size // 2:] = 2
    nuclei[nucleus_rows, :] = 3
    image = np.full((size, size), BACKGROUND + shift, dtype=np.uint16)
    image[cells > 0] = CELL
    image[nuclei > 0] = NUCLEUS
    second = image.copy()
    second[0, :] = np.iinfo(np.uint16).max
    stack = np.stack([image, second, cells, nuclei], axis=-1)
    return stack, cells, nuclei


def _write_plate(root):
    merged = root / 'merged'
    merged.mkdir(parents=True)
    names = []
    for well in ('plate1_A01', 'plate1_B02'):
        for number in (1, 2):
            stack, _, _ = _field(shift=number)
            name = f'{well}_{number}'
            np.save(merged / f'{name}.npy', stack)
            names.append(name)
    return merged, names


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        'src': str(merged), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3,
        'pathogen_mask_dim': None, 'cell_min_size': 0,
        'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'n_jobs': 1,
        'confluency': True, 'confluency_qc_threshold': 0.5,
    })
    settings.update(over)
    return settings


@pytest.fixture(scope='module')
def measured(tmp_path_factory):
    """One Measure run over four fields, after a mask stage wrote its counts."""
    from spacr.io import _save_object_counts_to_database

    root = tmp_path_factory.mktemp('image_table')
    merged, names = _write_plate(root)
    db = root / 'measurements' / 'measurements.db'
    db.parent.mkdir()
    masks = [np.load(merged / f'{name}.npy')[..., 2] for name in names]
    _save_object_counts_to_database(masks, 'cell', [f'{n}.npy' for n in names],
                                    str(db), '')
    measure.measure_crop(_settings(merged))
    return {'root': root, 'merged': merged, 'names': names, 'db': db}


def test_measure_writes_one_row_per_field_with_its_keys(measured):
    image = _table(measured['db'], 'image')
    assert sorted(image['file_name']) == sorted(measured['names'])
    assert image['file_name'].is_unique
    row = image.set_index('file_name').loc['plate1_B02_2']
    assert (row['plateID'], row['rowID'], row['columnID'], row['fieldID']) == (
        'plate1', 'r2', 'c2', 'f2')
    assert row['prcf'] == 'plate1_r2_c2_f2'
    assert row['cell_count'] == 2 and row['nucleus_count'] == 1
    assert row['cell_area_fraction'] == pytest.approx(26 / 64)
    assert row['nucleus_area_fraction'] == pytest.approx(19 / 64)
    assert 'cytoplasm_count' in image.columns


def test_background_median_excludes_every_mask(measured):
    image = _table(measured['db'], 'image').set_index('file_name')
    rescale = _table(measured['db'], 'intensity_rescale').set_index('file_name')
    for name in measured['names']:
        factor = rescale.loc[name, 'rescale_factor']
        shift = int(name[-1])
        expected = (BACKGROUND + shift) * factor
        assert image.loc[name, 'channel_0_background_median'] == pytest.approx(expected, rel=1e-3)
        assert image.loc[name, 'channel_0_background_p25'] == pytest.approx(expected, rel=1e-3)
        assert image.loc[name, 'rescale_factor'] == pytest.approx(factor)


def test_extras_cover_mean_percentiles_saturation_and_focus(measured):
    image = _table(measured['db'], 'image').set_index('file_name')
    row = image.loc['plate1_A01_1']
    for column in ('mean', 'p1', 'p99', 'saturated_fraction', 'focus'):
        assert f'channel_0_{column}' in image.columns
        assert f'channel_1_{column}' in image.columns
    assert row['channel_0_p1'] <= row['channel_0_mean'] <= row['channel_0_p99']
    assert row['channel_0_focus'] > 0
    assert row['channel_1_saturated_fraction'] >= row['channel_0_saturated_fraction']


def test_confluency_and_object_counts_are_in_the_same_row(measured):
    image = _table(measured['db'], 'image').set_index('file_name')
    confluency = _table(measured['db'], 'confluency').set_index('file_name')
    for name in measured['names']:
        assert image.loc[name, 'confluency'] == pytest.approx(confluency.loc[name, 'confluency'])
        assert image.loc[name, 'monolayer_ok'] == confluency.loc[name, 'monolayer_ok']
        assert image.loc[name, 'mask_cell_count'] == 2


def test_the_old_tables_still_read(measured):
    db = measured['db']
    assert len(measure._read_confluency(str(db))) == 4
    assert len(_table(db, 'intensity_rescale')) == 4
    counts = _table(db, 'object_counts')
    assert set(counts['count_type']) == {'cell'}
    assert len(counts) == 4


def test_a_reader_by_file_name_returns_the_background_median(measured):
    names = ['plate1_B02_1.npy', 'missing_field', 'plate1_A01_2']
    frame = _image_table_background(str(measured['db']), names, [0, 1])
    assert list(frame.index) == ['plate1_B02_1', 'missing_field', 'plate1_A01_2']
    assert list(frame.columns) == ['channel_0_background_median',
                                   'channel_1_background_median']
    assert np.isnan(frame.loc['missing_field']).all()
    image = _table(measured['db'], 'image').set_index('file_name')
    assert frame.loc['plate1_A01_2', 'channel_1_background_median'] == pytest.approx(
        image.loc['plate1_A01_2', 'channel_1_background_median'])
    absent = _image_table_background(str(measured['db']), ['plate1_A01_1'], [7])
    assert np.isnan(absent.iloc[0, 0])


def test_a_second_run_keeps_the_shared_table_in_place(measured, tmp_path):
    import shutil
    root = tmp_path / 'again'
    shutil.copytree(measured['root'], root)
    measure.measure_crop(_settings(root / 'merged'))
    tables = _tables(root / 'measurements' / 'measurements.db')
    assert 'image' in tables and 'image_1' not in tables and 'cell_1' in tables
    image = _table(root / 'measurements' / 'measurements.db', 'image')
    assert len(image) == 4 and (image['mask_cell_count'] == 2).all()


def test_background_excludes_every_mask_including_an_unmeasured_one():
    stack, cells, nuclei = _field()
    occupied = (cells > 0) | (nuclei > 0)
    values = _image_field_values(stack[..., :2], {'cell': cells}, occupied)
    assert values['channel_0_background_median'] == BACKGROUND
    union = _image_field_values(stack[..., :2], {'cell': cells, 'nucleus': nuclei})
    assert union['channel_0_background_median'] == BACKGROUND
    cell_only = _image_field_values(stack[..., :2], {'cell': cells})
    assert cell_only['channel_0_background_median'] > BACKGROUND
    assert union['channel_1_saturated_fraction'] == pytest.approx(1 / 64)
    assert union['channel_0_saturated_fraction'] == 0


def test_float_and_volumetric_fields():
    stack, cells, nuclei = _field()
    volume = np.stack([stack[..., :2], stack[..., :2]]).astype(np.float32)
    volume[0, 0, 0, 0] = np.nan
    masks = {'cell': np.stack([cells, cells]), 'nucleus': np.stack([nuclei, nuclei])}
    values = _image_field_values(volume, masks)
    assert values['channel_0_background_median'] == BACKGROUND
    assert values['channel_0_saturated_fraction'] is None
    assert values['cell_count'] == 2
    empty = _image_field_values(volume, {'all': np.ones(volume.shape[:-1], np.uint16)})
    assert np.isnan(empty['channel_0_background_median'])
    assert measure._image_object_count(np.zeros(0, np.uint8)) == 0


def test_focus_is_the_image_qc_laplacian_variance():
    from scipy.ndimage import laplace

    rng = np.random.default_rng(0)
    for plane in (rng.integers(0, 65535, (37, 41)).astype(np.uint16),
                  rng.integers(-300, 300, (9, 12)).astype(np.int16),
                  rng.random((20, 15)).astype(np.float32) * 1000,
                  np.full((5, 5), 7, dtype=np.uint8)):
        expected = float(np.var(laplace(plane.astype(np.float64))))
        assert measure._image_focus_variance(plane) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert all(np.isnan(_image_percentiles(np.array([], np.uint16), (1, 99))))


def test_large_labels_are_counted_without_a_histogram():
    mask = np.array([0, 1 << 25, 1 << 25, 3], dtype=np.int64)
    assert measure._image_object_count(mask) == 2


def test_writers_fill_their_own_columns_of_one_row(tmp_path):
    db = tmp_path / 'measurements' / 'measurements.db'
    assert _write_image_rows(db, []) == 0
    _write_image_rows(db, [{'file_name': 'Assay%5FPlate_A01_1.npy', 'a': 1.5,
                            'flag': True, 'blob': [1, 2]}])
    _write_image_rows(db, [{'file_name': 'Assay%5FPlate_A01_1', 'b': 'x',
                            'a': float('nan')}])
    _write_image_rows(db, [{'file_name': 'unparsed', 'c': np.int64(3), 'empty': None}])
    image = _read_image_table(str(db)).set_index('file_name')
    row = image.loc['Assay%5FPlate_A01_1']
    assert row['plateID'] == 'Assay_Plate' and row['prcf'] == 'Assay_Plate_r1_c1_f1'
    assert pd.isna(row['a']) and row['b'] == 'x' and row['flag'] == 1
    assert row['blob'] == '[1, 2]'
    assert image.loc['unparsed', 'c'] == 3 and pd.isna(image.loc['unparsed', 'plateID'])
    assert _read_image_table(str(tmp_path / 'none.db')).empty
    assert _image_table_background(str(tmp_path / 'none.db'), ['x'], [0]).shape == (1, 1)


def test_keys_carry_the_timepoint_for_a_timelapse():
    keys = _image_table_keys('plate1_A01_2_5.npy', timelapse=True)
    assert keys['timeID'] == 't5' and keys['prcf'] == 'plate1_r1_c1_f2_t5'
    assert _image_table_keys('plate1_A01_2_5')['timeID'] is None


def test_a_queued_measure_packet_carries_the_image_row(tmp_path):
    from spacr.database_concurrency import _capture_write_packet

    db = tmp_path / 'measurements' / 'measurements.db'
    db.parent.mkdir()
    with _capture_write_packet() as operations:
        measure._write_image_record(str(tmp_path), 'plate1_A01_1', {},
                                    {'channel_0_background_median': 4.0})
    assert [operation[0] for operation in operations] == ['image']
    assert not db.exists()
    assert measure._commit_measure_packet(
        str(db), {'ticket': 't1', 'field': 'plate1_A01_1', 'operations': operations})
    frame = _image_table_background(str(db), ['plate1_A01_1'], [0])
    assert frame.iloc[0, 0] == 4.0


def test_image_screening_writes_its_verdicts_to_image(tmp_path):
    from spacr.image_quality import ensure_no_retained_measurements, screen_fields

    stack = tmp_path / 'stack'
    stack.mkdir()
    sharp = np.zeros((32, 32), dtype=np.uint16)
    sharp[::2, ::2] = 4000
    np.save(stack / 'plate1_A01_1.npy', sharp)
    np.save(stack / 'plate1_A01_2.npy', np.full((32, 32), 7, dtype=np.uint16))
    policy = dict(image_qc_mode='exclude', image_qc_min_focus={0: 1.})
    assert screen_fields(tmp_path, policy) == ['plate1_A01_2.npy']
    db = tmp_path / 'measurements' / 'measurements.db'
    image = _read_image_table(str(db)).set_index('file_name')
    assert image.loc['plate1_A01_2', 'image_qc_status'] == 'excluded_image_quality'
    assert 'focus_below_threshold' in image.loc['plate1_A01_2', 'image_qc_reasons']
    assert image.loc['plate1_A01_1', 'image_qc_status'] == 'accepted'
    assert image.loc['plate1_A01_1', 'channel_0_qc_focus_variance'] > 1
    assert image.loc['plate1_A01_1', 'plateID'] == 'plate1'
    assert (tmp_path / 'qc' / 'image_quality.csv').is_file()
    ensure_no_retained_measurements(tmp_path, ['plate1_A01_2.npy'])
    assert screen_fields(tmp_path, policy) == ['plate1_A01_2.npy']
