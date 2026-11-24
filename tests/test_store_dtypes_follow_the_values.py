"""The recorded dtype map is the STARTING choice; the part's own values decide.

float16 holds integers exactly only up to 2048, so a bbox area of 5000 went
through the map's float16 and came back off by up to 4. Every narrowed float is
checked against the values of the part it is written with.
"""
import json
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from spacr import store


def _frame(n=200, **columns):
    frame = pd.DataFrame({'object_key': np.arange(n, dtype=np.int64)})
    for name, values in columns.items():
        frame[name] = values
    return frame


@pytest.fixture
def narrow_map(tmp_path, monkeypatch):
    """A map that starts every test column at float16."""
    names = ['nucleus_area_bbox', 'nucleus_noise', 'nucleus_small_ints',
             'nucleus_smooth', 'nucleus_flags']
    path = tmp_path / 'map.json'
    path.write_text(json.dumps({'version': 1, 'columns': {
        n: {'kind': 'float', 'dtype': 'float16'} for n in names}}))
    monkeypatch.setattr(store, '_DTYPE_MAP_PATH', str(path))
    store._read_dtype_map.cache_clear()
    yield
    store._read_dtype_map.cache_clear()


def _write(tmp_path, frame):
    manifest = store.write_table(str(tmp_path / 'proj'), 'nucleus', frame,
                                 plate='p', well='A1', narrow=True)
    store.register(str(tmp_path / 'proj'), run={'run_id': 'r', 'settings': {}})
    return manifest


def _catalog(tmp_path, columns="path, promoted"):
    """The registered parquet parts of the nucleus table, from the catalog."""
    conn = duckdb.connect(os.path.join(str(tmp_path / 'proj'), store.CATALOG),
                          read_only=True)
    try:
        return conn.execute(
            f"SELECT {columns} FROM spacr_files WHERE kind = 'parquet' "
            f"AND table_name = 'nucleus'").fetchall()
    finally:
        conn.close()


def _stored(tmp_path, column):
    import pyarrow.parquet as pq
    (path, _promoted), = _catalog(tmp_path)
    return pq.read_schema(os.path.join(str(tmp_path / 'proj'),
                                       path)).field(column).type


def test_integer_valued_areas_above_2048_stay_exact(tmp_path, narrow_map):
    rng = np.random.default_rng(1)
    areas = rng.integers(2049, 9000, 200).astype('float32')
    manifest = _write(tmp_path, _frame(nucleus_area_bbox=areas))
    assert str(_stored(tmp_path, 'nucleus_area_bbox')) == 'float'
    with store.open(str(tmp_path / 'proj')) as opened:
        found = opened.query(table='nucleus',
                             features=['nucleus_area_bbox'])
    assert np.array_equal(np.sort(found['nucleus_area_bbox'].to_numpy()),
                          np.sort(areas))
    entry = {p['column']: p for p in manifest['promoted']}['nucleus_area_bbox']
    assert (entry['recorded'], entry['stored']) == ('float16', 'float32')
    assert 'integers' in entry['reason'] or 'range' in entry['reason'] \
        or 'rounding' in entry['reason']


def test_a_float_column_too_coarse_for_float16_is_promoted(tmp_path, narrow_map):
    rng = np.random.default_rng(2)
    noise = (1000.0 + rng.normal(0, 0.05, 200)).astype('float32')
    manifest = _write(tmp_path, _frame(nucleus_noise=noise))
    entry = {p['column']: p for p in manifest['promoted']}['nucleus_noise']
    assert entry['stored'] == 'float32'
    assert 'SD' in entry['reason']


def test_a_column_that_passes_stays_float16(tmp_path, narrow_map):
    small = np.tile(np.arange(0, 20, dtype='float32'), 10)
    smooth = np.linspace(0, 1, 200, dtype='float32')
    manifest = _write(tmp_path, _frame(nucleus_small_ints=small,
                                       nucleus_smooth=smooth))
    assert manifest['promoted'] == []
    assert str(_stored(tmp_path, 'nucleus_small_ints')) == 'halffloat'
    assert str(_stored(tmp_path, 'nucleus_smooth')) == 'halffloat'


def test_nan_and_inf_are_preserved(tmp_path, narrow_map):
    values = np.linspace(0, 1, 200, dtype='float32')
    values[3], values[4], values[5] = np.nan, np.inf, -np.inf
    _write(tmp_path, _frame(nucleus_smooth=values))
    with store.open(str(tmp_path / 'proj')) as opened:
        found = opened.query(table='nucleus', features=['nucleus_smooth'])
    got = found['nucleus_smooth'].to_numpy()
    assert np.isnan(got).sum() == 1 and np.isposinf(got).sum() == 1
    assert np.isneginf(got).sum() == 1


def test_the_promotions_are_recorded_in_the_catalog_per_part(tmp_path, narrow_map):
    areas = np.arange(3000, 3200, dtype='float32')
    _write(tmp_path, _frame(nucleus_area_bbox=areas,
                            nucleus_smooth=np.linspace(0, 1, 200,
                                                       dtype='float32')))
    rows = _catalog(tmp_path, "plate, well, promoted")
    assert len(rows) == 1
    promoted = json.loads(rows[0][2])
    assert [p['column'] for p in promoted] == ['nucleus_area_bbox']
    assert promoted[0]['stored'] == 'float32'


def test_the_same_input_gives_the_same_choice(tmp_path, narrow_map):
    areas = np.arange(3000, 3200, dtype='float32')
    first = _write(tmp_path, _frame(nucleus_area_bbox=areas))['promoted']
    second = store.write_table(str(tmp_path / 'proj2'), 'nucleus',
                               _frame(nucleus_area_bbox=areas), plate='p',
                               well='A1', narrow=True)['promoted']
    assert first == second
