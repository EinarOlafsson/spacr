"""Unified store, step S4: Measure and Make Masks write the spaCR store natively.

measurements.db is no longer written; every reader opens the store through
spacr.store.open, spacr.tabular or spacr.measurement_store.connect, and an
older project that still has measurements.db keeps reading it.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measurement_store as ms
from spacr import store


def _field(field, n, column='c1', extra=False):
    frame = pd.DataFrame({
        'object_label': np.arange(1, n + 1), 'plateID': 'plate1',
        'rowID': 'r1', 'columnID': column, 'fieldID': f'f{field}',
        'prcf': f'plate1_r1_{column}_f{field}',
        'file_name': f'plate1_r1{column}_{field}',
        'cell_area': np.linspace(100.0, 900.0, n)})
    frame['prcfo'] = frame['prcf'] + '_o' + frame['object_label'].astype(str)
    if extra:
        frame['cell_extra'] = 1.5
    return frame


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'measurements' / 'measurements.db'
    ms.ensure_store(db)
    yield str(db)
    ms.forget_sessions()


def _spool(db, fields):
    for number, column, extra in fields:
        with ms.field_packet() as packet:
            ms.append(db, 'cell', _field(number, 3, column, extra))
            ms.append(db, 'image', pd.DataFrame([{
                'file_name': f'plate1_r1{column}_{number}', 'plateID': 'plate1',
                'rowID': 'r1', 'columnID': column, 'measured': number}]))
        ms.spool_field(db, f'plate1_r1{column}_{number}.npy', packet,
                       ticket=f't{number}', run_id='run')


def test_fields_spool_then_compact_into_one_schema_per_table(project):
    _spool(project, [(2, 'c1', False), (1, 'c1', False), (3, 'c2', True)])
    assert ms.measured_fields(project) == [
        'plate1_r1c1_1.npy', 'plate1_r1c1_2.npy', 'plate1_r1c2_3.npy']
    report = ms.compact(project, run_id='run', settings={'a': 1})
    assert report['tables']['cell'] == 9
    cells = ms.read(project, 'cell')
    assert list(cells['prcfo'])[:3] == [f'plate1_r1_c1_f1_o{i}' for i in (1, 2, 3)]
    assert cells['cell_extra'].isna().sum() == 6
    assert not os.path.exists(project)
    assert not os.path.isdir(ms.spool_of(project))
    assert ms.measured_fields(project) == [
        'plate1_r1c1_1.npy', 'plate1_r1c1_2.npy', 'plate1_r1c2_3.npy']
    with store.open(os.path.dirname(project)) as opened:
        assert opened.count('cell') == 9


def test_an_upsert_table_merges_columns_by_key(project):
    folder = os.path.dirname(project)
    ms.write_rows(folder, 'image', pd.DataFrame([{
        'file_name': 'plate1_r1c1_1', 'plateID': 'plate1', 'rowID': 'r1',
        'columnID': 'c1', 'mask_cell_count': 7}]))
    _spool(project, [(1, 'c1', False)])
    ms.compact(project, run_id='run')
    image = ms.read(project, 'image')
    assert len(image) == 1
    assert image['mask_cell_count'].iloc[0] == 7
    assert image['measured'].iloc[0] == 1


def test_annotations_go_through_the_store_connection(project):
    folder = os.path.dirname(project)
    frame = pd.concat([_field(1, 2), _field(2, 2, 'c2')]).assign(
        png_path=list('abcd'))
    ms.write_rows(folder, 'png_list', frame)
    from spacr.database_concurrency import connect
    conn = connect(project)
    assert isinstance(conn, sqlite3.Connection)
    conn.execute('ALTER TABLE png_list ADD COLUMN label INTEGER')
    conn.execute("UPDATE png_list SET label = 2 WHERE png_path = 'c'")
    conn.commit()
    conn.close()
    assert not os.path.exists(project)
    ms.forget_sessions()
    labels = ms.read(project, 'png_list').set_index('png_path')['label']
    assert labels['c'] == 2 and labels.drop('c').isna().all()
    read = pd.read_sql_query('SELECT png_path, label FROM png_list ORDER BY rowid',
                             ms.connect(project, readonly=True))
    assert len(read) == 4


def test_object_tables_are_read_only_through_sql(project):
    folder = os.path.dirname(project)
    ms.write_rows(folder, 'cell', _field(1, 2))
    conn = ms.connect(project)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute('UPDATE cell SET cell_area = 0')
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ('cell',) in rows


def test_tabular_and_the_legacy_reader_open_the_store(project):
    folder = os.path.dirname(project)
    ms.write_rows(folder, 'cell', _field(1, 2))
    from spacr.io import _read_db
    from spacr.tabular import read_table
    assert len(read_table(project, table='cell', report=None)) == 2
    assert 'object_key' not in _read_db(project, ['cell'])[0].columns


def test_an_older_project_keeps_its_measurements_db(tmp_path):
    db = tmp_path / 'measurements' / 'measurements.db'
    db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        _field(1, 2).to_sql('cell', conn, index=False)
    assert not ms.store_native(db)
    assert len(ms.read(db, 'cell')) == 2
    archived = ms.retire_legacy_database(db)
    assert archived and not db.exists()
    assert len(ms.read(db, 'cell')) == 2


def test_retired_measurement_backend_values_still_load():
    from spacr.settings import _retire_measurement_backend
    from spacr.validate import _check_retired_values
    for value in ('sqlite', 'duckdb', 'parquet', 'postgres'):
        settings = {'measurement_backend': value}
        assert _check_retired_values(settings)
        _retire_measurement_backend(settings)
        assert settings['measurement_backend'] == 'spacr_store'
    settings = {'measurement_backend': 'postgres'}
    _retire_measurement_backend(settings)
    assert settings['_measurement_backend_requested'] == 'postgres'
