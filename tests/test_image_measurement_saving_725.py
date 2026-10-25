"""Item 725 round 3: one image-level table in measurements.db, saved idempotently and exported."""
from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import CancelledError

import numpy as np
import pandas as pd
import pytest

from spacr import schema
from spacr.measure import (
    _IMAGE_MEASUREMENT_KEYS,
    _export_image_measurements,
    _image_level_features,
    _image_measurement_export_formats,
    _image_measurement_table,
    _publish_image_measurements,
    _set_image_annotations,
    _write_image_measurements,
)


def _row(source, well='A01', field='1', groups=('intensity',), value=0.0):
    path = source / f'plate1_{well}_{field}.tif'
    row = dict(plate='plate1', well=well, row=well[0], column=int(well[1:]), field=field,
               field_id=field, time_id='', z_id='', plateID='plate1', rowID='r1',
               columnID=f'c{int(well[1:])}', fieldID=schema.field_id(field), timeID='', sliceID='',
               source=str(source), source_files=json.dumps([['0', str(path)]]), status='ok',
               condition='', image_features=','.join(groups), time_index=None, z_mode='plane',
               z_index=None)
    planes = {0: np.full((16, 16), 100.0 + value) + np.arange(256).reshape(16, 16)}
    row.update(_image_level_features(planes, groups=groups, confluency=False))
    return row


def _stored(db):
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(item) for item in connection.execute('SELECT * FROM fov')]


def test_rows_are_keyed_like_image_annotations_and_reruns_never_duplicate(tmp_path):
    db = tmp_path / 'measurements' / 'measurements.db'
    first = _write_image_measurements(db, [_row(tmp_path), _row(tmp_path, 'B02')])
    assert (first['inserted'], first['updated']) == (2, 0)
    again = _write_image_measurements(db, [_row(tmp_path, value=5.0)])
    assert (again['inserted'], again['updated']) == (0, 1)
    rows = {row['wellID']: row for row in _stored(db)}
    assert len(rows) == 2
    assert rows['A01']['file'] == 'plate1_A01_1.tif'
    assert rows['A01']['fieldID'] == 'f1' and rows['A01']['plateID'] == 'plate1'
    assert rows['A01']['channel_0_intensity_min'] == 105.0
    assert rows['B02']['channel_0_intensity_min'] == 100.0
    assert _IMAGE_MEASUREMENT_KEYS == schema.IMAGE_MEASUREMENT_INDEX_COLUMNS
    with sqlite3.connect(db) as connection:
        created = connection.execute("SELECT sql FROM sqlite_master WHERE name = 'fov'").fetchone()[0]
        indexed = [row[2] for row in connection.execute("PRAGMA index_info('fov_image_key')")]
    assert 'WITHOUT ROWID' in created.upper()
    assert tuple(indexed) == _IMAGE_MEASUREMENT_KEYS


def test_changing_the_selection_adds_columns_and_keeps_earlier_values(tmp_path):
    db = tmp_path / 'measurements.db'
    _write_image_measurements(db, [_row(tmp_path, groups=('intensity',))])
    report = _write_image_measurements(db, [_row(tmp_path, groups=('focus',))])
    assert 'channel_0_focus_brenner' in report['columns_added']
    row, = _stored(db)
    assert row['channel_0_intensity_mean'] is not None
    assert row['channel_0_focus_brenner'] is not None
    assert row['image_features'] == 'intensity,focus'


def test_a_cancelled_or_failed_save_leaves_the_table_unchanged(tmp_path):
    db = tmp_path / 'measurements.db'
    _write_image_measurements(db, [_row(tmp_path)])
    before = _stored(db)
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        _write_image_measurements(db, [_row(tmp_path, 'C03', groups=('texture',))], cancelled=event)
    with pytest.raises(ValueError, match='fieldID'):
        _write_image_measurements(db, [dict(plateID='plate1', well='A01')])
    assert _stored(db) == before


def test_a_legacy_rowid_table_is_rebuilt_without_losing_rows(tmp_path):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE fov (field TEXT, legacy_value INTEGER)')
        connection.execute("INSERT INTO fov VALUES ('legacy', 42)")
    _write_image_measurements(db, [_row(tmp_path)])
    rows = _stored(db)
    legacy, = (row for row in rows if row['field'] == 'legacy')
    assert legacy['legacy_value'] == 42 and legacy['field_uid'] == 'legacy-rowid-1'
    assert len(rows) == 2


def test_the_ready_to_compare_table_joins_annotations_and_is_exported(tmp_path):
    db = tmp_path / 'measurements' / 'measurements.db'
    fields = [dict(plate='plate1', well=well, row=well[0], column=int(well[1:]), field='1',
                   files=(('0', str(tmp_path / f'plate1_{well}_1.tif')),)) for well in ('A01', 'B02')]
    _set_image_annotations(db, fields[:1], 'cell_type', 'HeLa', source=tmp_path)
    _set_image_annotations(db, fields[1:], 'cell_type', 'HFF', source=tmp_path)
    report = _publish_image_measurements(db, [_row(tmp_path), _row(tmp_path, 'B02', value=50.0)],
                                         dict(overview_measure_export_format='both'))
    assert set(report['exports']) == {'csv', 'xlsx'}
    table = _image_measurement_table(db)
    assert list(table.columns[:4]) == list(_IMAGE_MEASUREMENT_KEYS)
    assert table.set_index('wellID')['cell_type'].to_dict() == {'A01': 'HeLa', 'B02': 'HFF'}
    measured = [name for name in table.columns if name.startswith('channel_')]
    assert table.columns.get_loc('cell_type') < table.columns.get_loc(measured[0])
    exported = pd.read_csv(report['exports']['csv'])
    assert report['exports']['csv'].endswith('measurements/image_measurements.csv')
    assert len(exported) == 2 and list(exported.columns) == list(table.columns)
    assert len(pd.read_excel(report['exports']['xlsx'])) == 2
    _publish_image_measurements(db, [_row(tmp_path, value=1.0)], dict(overview_measure_export_format='csv'))
    assert len(pd.read_csv(report['exports']['csv'])) == 2


def test_export_formats_follow_the_setting(tmp_path):
    assert _image_measurement_export_formats() == ('csv',)
    assert _image_measurement_export_formats(dict(overview_measure_export_format='none')) == ()
    assert _image_measurement_export_formats(dict(overview_measure_export_format='xlsx')) == ('xlsx',)
    with pytest.raises(ValueError):
        _image_measurement_export_formats(dict(overview_measure_export_format='pdf'))
    assert _export_image_measurements(tmp_path / 'missing.db') == {}
    with pytest.raises(ValueError):
        _export_image_measurements(tmp_path / 'missing.db', formats=('json',))
    report = _publish_image_measurements(tmp_path / 'measurements.db', [_row(tmp_path)],
                                         dict(overview_measure_export_format='none'))
    assert report['exports'] == {} and report['inserted'] == 1
    assert not (tmp_path / 'image_measurements.csv').exists()


def test_identity_of_rows_from_other_callers(tmp_path):
    from spacr.measure import _image_measurement_identity
    inside = _image_measurement_identity(dict(plate='p', well='A01', fieldID='f1', source=str(tmp_path),
                                              source_files=[['0', str(tmp_path / 'sub' / 'a.tif')]],
                                              flag=np.bool_(True), count=np.int64(3), axes=[1, 2]))
    assert inside['plateID'] == 'p' and inside['wellID'] == 'A01' and inside['file'] == 'sub/a.tif'
    assert (inside['flag'], inside['count'], inside['axes']) == (True, 3, '[1, 2]')
    outside = _image_measurement_identity(dict(plateID='p', wellID='A01', fieldID='f1', source=str(tmp_path / 'x'),
                                               source_files=json.dumps([['0', str(tmp_path / 'a.tif')]])))
    assert outside['file'] == str((tmp_path / 'a.tif').resolve())
    given = _image_measurement_identity(dict(plateID='p', wellID='A01', fieldID='f1', file='a.tif', field_uid='u1'))
    assert given['file'] == 'a.tif' and given['field_uid'] == 'u1'
    bare = _image_measurement_identity(dict(plateID='p', wellID='A01', fieldID='f1'))
    assert bare['file'] == '' and len(bare['field_uid']) == 64


def test_nothing_to_save_and_rows_without_a_feature_list(tmp_path):
    db = tmp_path / 'measurements.db'
    assert _write_image_measurements(db, [])['inserted'] == 0 and not db.exists()
    row = dict(plateID='p', wellID='A01', fieldID='f1', file='a.tif', channel_0_intensity_mean=1.0)
    _write_image_measurements(db, [row])
    report = _write_image_measurements(db, [dict(row, channel_0_intensity_mean=2.0)])
    assert report['updated'] == 1
    stored, = _stored(db)
    assert stored['channel_0_intensity_mean'] == 2.0
