"""Fresh Measure populations retain old tables without duplicating canonical rows."""
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from spacr import measure
from spacr.cancellation import PipelineCancelled
from spacr.errors import ConfigurationError
from spacr.schema import ORGANELLE_ROLES
from tests.test_measure_crop_core_synth import (
    FIXTURE_COUNTS, _build_merged_stack, _settings_for, _write_stack,
)


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _project(tmp_path):
    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir()
    np.save(merged / 'plate1_A01_F001.npy', np.zeros((4, 4, 7), np.uint16))
    return merged, tmp_path / 'measurements' / 'measurements.db'


def _seed(database, tables=('cell', 'nucleus', 'png_list')):
    with sqlite3.connect(database) as connection:
        for index, table in enumerate(tables):
            connection.execute(f'CREATE TABLE {_quote(table)} (object_label INTEGER, value INTEGER)')
            connection.execute(f'INSERT INTO {_quote(table)} VALUES (1, ?)', (index + 10,))


def _snapshot(database):
    with sqlite3.connect(database) as connection:
        schema = connection.execute(
            "SELECT name,type,sql FROM sqlite_master ORDER BY name").fetchall()
        rows = {name: connection.execute(f'SELECT * FROM {_quote(name)}').fetchall()
                for name, kind, _sql in schema if kind == 'table'}
    return schema, rows


def test_real_three_measure_runs_archive_populations_and_keep_png_and_organelle_identity(
        tmp_path, synth_masks_multi, rng):
    data = _build_merged_stack(synth_masks_multi, rng, with_organelle=True)
    merged, _name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, resume=False, database_write_queue_gib=0,
                             organelle_mask_dim=7, organelle_min_area=0)
    database = tmp_path / 'measurements' / 'measurements.db'
    source = (merged / 'plate1_A01_F001.npy').read_bytes()
    measure.measure_crop(dict(settings))
    _first_schema, first = _snapshot(database)
    expected = dict(FIXTURE_COUNTS, png_list=4)
    organs = [table for table in ORGANELLE_ROLES if table in first and first[table]]
    assert organs
    for table in organs:
        expected[table] = len(first[table])
    assert {table: len(first[table]) for table in expected} == expected
    with sqlite3.connect(database) as connection:
        first_pngs = connection.execute('SELECT png_path,prcfo,cell_id FROM png_list ORDER BY prcfo').fetchall()
    png_identity = [row[1:] for row in first_pngs]
    png_names = [Path(row[0]).name for row in first_pngs]
    saved_pngs = {Path(row[0]): Path(row[0]).read_bytes() for row in first_pngs}
    previous = first
    for suffix in (1, 2):
        measure.measure_crop(dict(settings))
        _schema, current = _snapshot(database)
        for table, count in expected.items():
            assert len(current[table]) == count
            assert current[f'{table}_{suffix}'] == previous[table]
            assert current[f'{table}_1'] == first[table]
        with sqlite3.connect(database) as connection:
            labels = connection.execute(
                'SELECT object_label,COUNT(*) FROM cell GROUP BY object_label').fetchall()
            pngs = connection.execute('SELECT png_path,prcfo,cell_id FROM png_list ORDER BY prcfo').fetchall()
        assert labels == [(1, 1), (2, 1), (3, 1), (4, 1)]
        assert [row[1:] for row in pngs] == png_identity
        assert [Path(row[0]).name for row in pngs] == png_names
        current_paths = {Path(row[0]) for row in pngs}
        assert len(current_paths) == 4 and not current_paths.intersection(saved_pngs)
        for path, pixels in saved_pngs.items():
            assert path.read_bytes() == pixels
        saved_pngs.update({path: path.read_bytes() for path in current_paths})
        previous = current
        assert (merged / 'plate1_A01_F001.npy').read_bytes() == source


def test_archive_collision_uses_common_suffix_preserves_old_archives_and_provenance(tmp_path):
    merged, database = _project(tmp_path)
    protected = ('conversion_map', 'align_coordinates', 'foreign_cell', 'object_counts',
                 'pivoted_counts', 'settings_history', 'fov', 'image_annotations',
                 'image_annotations_meta', 'image_annotations_history',
                 'image_comparison_groups', '_spacr_write_queue_commits')
    dynamic = ORGANELLE_ROLES[4]
    live = ('cell', 'nucleus', 'png_list', dynamic, 'custom_derived')
    _seed(database, live + protected + ('cell_1', 'png_list_2'))
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE empty_derived (value TEXT)')
    before_schema, before = _snapshot(database)
    settings = {'src': str(merged)}
    plan = measure._measurement_run_archive_plan(settings)
    assert len(plan) == 1
    assert plan[0]['tables'] == {table: f'{table}_3' for table in (*live, 'empty_derived')}
    assert _snapshot(database) == (before_schema, before)
    measure._archive_measurement_run(settings)
    _schema, current = _snapshot(database)
    for table in (*live, 'empty_derived'):
        assert table not in current
        assert current[f'{table}_3'] == before[table]
    for table in (*protected, 'cell_1', 'png_list_2'):
        assert current[table] == before[table]
    assert measure._archive_measurement_run(settings) == []


def test_archive_keeps_existing_views_on_old_population_and_releases_index_names(tmp_path):
    merged, database = _project(tmp_path)
    _seed(database)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE UNIQUE INDEX cell_label_index ON cell(object_label)')
        connection.execute('CREATE VIEW old_summary AS SELECT SUM(value) AS total FROM cell')
        connection.execute('CREATE VIEW old_report AS SELECT total FROM old_summary')
    measure._archive_measurement_run({'src': str(merged)})
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT * FROM old_summary').fetchall() == [(10,)]
        assert connection.execute('SELECT * FROM old_report').fetchall() == [(10,)]
        connection.execute('CREATE TABLE cell (object_label INTEGER, value INTEGER)')
        connection.execute('CREATE UNIQUE INDEX cell_label_index ON cell(object_label)')
        connection.execute('INSERT INTO cell VALUES (1,99)')
        assert connection.execute('SELECT * FROM old_report').fetchall() == [(10,)]


def test_cancellation_after_renames_rolls_back_every_table_and_index(tmp_path, monkeypatch):
    from spacr import cancellation

    merged, database = _project(tmp_path)
    _seed(database)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE INDEX cell_value ON cell(value)')
    before = _snapshot(database)
    calls = []

    def stop():
        calls.append(True)
        if len(calls) == 2:
            raise PipelineCancelled('cancel during archive transaction')

    monkeypatch.setattr(cancellation, 'checkpoint', stop)
    with pytest.raises(PipelineCancelled, match='archive transaction'):
        measure._archive_measurement_run({'src': str(merged)})
    assert len(calls) == 2
    assert _snapshot(database) == before


@pytest.mark.parametrize('skip', [{'resume': True}, {'dry_run': True},
                                 {'save_measurements': False, 'save_png': False}])
def test_continuation_and_nonwriting_modes_leave_existing_tables_unchanged(tmp_path, skip):
    merged, database = _project(tmp_path)
    _seed(database)
    before = _snapshot(database)
    settings = dict(src=str(merged), **skip)
    assert measure._measurement_run_archive_plan(settings) == []
    assert measure._archive_measurement_run(settings) == []
    assert _snapshot(database) == before


def test_empty_canonical_tables_do_not_trigger_archiving(tmp_path):
    merged, database = _project(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (object_label INTEGER)')
        connection.execute('CREATE TABLE png_list (png_path TEXT)')
    before = _snapshot(database)
    assert measure._archive_measurement_run({'src': str(merged)}) == []
    assert _snapshot(database) == before


def test_invalid_real_measure_settings_preserve_populated_previous_tables(
        tmp_path, synth_masks_multi, rng):
    merged, _name = _write_stack(tmp_path, _build_merged_stack(synth_masks_multi, rng))
    database = tmp_path / 'measurements' / 'measurements.db'
    _seed(database)
    before = _snapshot(database)
    settings = _settings_for(merged, normalize=[1, 99], normalize_by='invalid')
    measure.measure_crop(settings)
    assert _snapshot(database) == before


def test_run_locks_refuse_overlap_and_release_on_exception_and_partial_acquisition(tmp_path):
    first, _db = _project(tmp_path / 'a')
    second, _other = _project(tmp_path / 'b')
    with measure._measurement_run_locks([str(second)]):
        with pytest.raises(ConfigurationError, match='Another Measure run'):
            with measure._measurement_run_locks([str(first), str(second)]):
                pytest.fail('overlapping Measure runs were allowed')
        with measure._measurement_run_locks([str(first), str(first.parent)]):
            pass
    with pytest.raises(RuntimeError, match='leave run'):
        with measure._measurement_run_locks([str(first)]):
            raise RuntimeError('leave run')
    with measure._measurement_run_locks([str(first), str(second)]):
        pass


def test_archive_preserves_watch_receipts_and_independent_saved_condition_tables(tmp_path):
    import pandas as pd
    from spacr.condition_annotations import (
        apply_conditions, new_definition, save_annotated_table, source_context,
    )

    merged, database = _project(tmp_path)
    _seed(database, ('cell',))
    source = source_context(str(database), 'cell')
    base = pd.DataFrame({'wellID': ['A1', 'B1'], 'value': [4, 7]})
    definition = new_definition(base, source)
    definition['conditions'] = [dict(name='treated', metadata_column='wellID',
                                    include='^A', exclude='', manual_rows=[])]
    frame = apply_conditions(base, definition, source)
    save_annotated_table(database, 'saved_treatments', frame, definition, source)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE spacr_watch_fields (field TEXT PRIMARY KEY, merged_at REAL)')
        connection.execute('INSERT INTO spacr_watch_fields VALUES (?, ?)', ('plate1_A01_F001', 123.5))
        payload = connection.execute(
            'SELECT payload_json FROM _spacr_condition_annotations WHERE table_name=?',
            ('saved_treatments',)).fetchone()[0]
        connection.execute('INSERT INTO _spacr_condition_annotations VALUES (?, ?)', ('cell', payload))
    before_schema, before = _snapshot(database)
    plan = measure._measurement_run_archive_plan({'src': str(merged)})
    assert plan[0]['tables'] == {'cell': 'cell_1'}
    measure._archive_measurement_run({'src': str(merged)})
    _schema, current = _snapshot(database)
    for table in ('spacr_watch_fields', 'saved_treatments'):
        assert current[table] == before[table]
    assert dict(current['_spacr_condition_annotations']) == {
        ('cell_1' if table == 'cell' else table): payload
        for table, payload in before['_spacr_condition_annotations']}
    assert current['cell_1'] == before['cell']
    assert 'cell' not in current


def test_archive_rebinds_canonical_annotation_receipt_and_rolls_back_malformed_payload(tmp_path):
    import json
    import pandas as pd
    from spacr.condition_annotations import (
        apply_conditions, new_definition, save_annotated_table,
        saved_table_annotation, source_context,
    )

    merged, database = _project(tmp_path)
    base = pd.DataFrame({'wellID': ['A1', 'B1'], 'value': [4, 7]})
    with sqlite3.connect(database) as connection:
        base.to_sql('foreign_input', connection, index=False)
    source = source_context(str(database), 'foreign_input')
    definition = new_definition(base, source)
    definition['conditions'] = [dict(name='treated', metadata_column='wellID',
                                    include='^A', exclude='', manual_rows=[])]
    frame = apply_conditions(base, definition, source)
    for table in ('cell', 'saved_treatments'):
        save_annotated_table(database, table, frame, definition, source)
    with sqlite3.connect(database) as connection:
        original_receipts = dict(connection.execute(
            'SELECT table_name,payload_json FROM _spacr_condition_annotations'))
    original = json.loads(original_receipts['cell'])
    measure._archive_measurement_run({'src': str(merged)})
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (value INTEGER)')
        connection.execute('INSERT INTO cell VALUES (99)')
        archived = pd.read_sql_query('SELECT * FROM cell_1', connection)
        receipts = dict(connection.execute(
            'SELECT table_name,payload_json FROM _spacr_condition_annotations'))
    assert set(receipts) == {'cell_1', 'saved_treatments'}
    assert receipts['saved_treatments'] == original_receipts['saved_treatments']
    expected = json.loads(original_receipts['cell'])
    expected['editable_definition']['source']['table'] = 'cell_1'
    assert json.loads(receipts['cell_1']) == expected
    assert expected['source'] == original['source']
    assert expected['original_definition'] == original['original_definition']
    editable = saved_table_annotation(database, 'cell_1', archived)
    archived_source = source_context(str(database), 'cell_1')
    unannotated = archived.drop(columns=['condition'])
    reproduced = apply_conditions(unannotated, editable, archived_source)
    assert reproduced.fillna('').to_dict('list') == archived.fillna('').to_dict('list')
    assert saved_table_annotation(database, 'cell', pd.DataFrame({'value': [99]})) is None
    editable['conditions'][0]['include'] = '^B'
    edited = apply_conditions(unannotated, editable, archived_source)
    save_annotated_table(database, 'edited_conditions', edited, editable, archived_source)
    with sqlite3.connect(database) as connection:
        saved = pd.read_sql_query('SELECT * FROM edited_conditions', connection)
    assert saved_table_annotation(database, 'edited_conditions', saved) is not None
    for malformed in ('{', '[]', '{}', '{"editable_definition":{"source":null}}'):
        with sqlite3.connect(database) as connection:
            connection.execute('INSERT OR REPLACE INTO _spacr_condition_annotations VALUES (?,?)',
                               ('cell', malformed))
        before = _snapshot(database)
        with pytest.raises(ValueError):
            measure._archive_measurement_run({'src': str(merged)})
        assert _snapshot(database) == before


def test_archive_trigger_names_are_collision_safe_and_stay_on_the_old_population(tmp_path):
    merged, database = _project(tmp_path)
    _seed(database, ('cell',))
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE _spacr_trigger_events (value INTEGER)')
        connection.execute('CREATE INDEX cell_value ON cell(value)')
        connection.execute('CREATE INDEX cell_value__cell_1 ON _spacr_trigger_events(value)')
        connection.execute('CREATE TRIGGER cell_changed AFTER UPDATE ON cell '
                           'BEGIN INSERT INTO _spacr_trigger_events VALUES (NEW.value); END')
        connection.execute('CREATE TRIGGER cell_changed__cell_1 AFTER UPDATE ON _spacr_trigger_events '
                           'BEGIN SELECT 1; END')
    measure._archive_measurement_run({'src': str(merged)})
    with sqlite3.connect(database) as connection:
        triggers = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='cell_1'").fetchall()
        assert len(triggers) == 1
        assert triggers[0][0] not in ('cell_changed', 'cell_changed__cell_1')
        indexes = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cell_1'").fetchall()
        assert len(indexes) == 1
        assert indexes[0][0] not in ('cell_value', 'cell_value__cell_1')
        connection.execute('CREATE TABLE cell (object_label INTEGER, value INTEGER)')
        connection.execute('CREATE INDEX cell_value ON cell(value)')
        connection.execute('INSERT INTO cell VALUES (1,99)')
        connection.execute('UPDATE cell SET value=100')
        assert connection.execute('SELECT * FROM _spacr_trigger_events').fetchall() == []
        connection.execute('UPDATE cell_1 SET value=42')
        assert connection.execute('SELECT * FROM _spacr_trigger_events').fetchall() == [(42,)]
        connection.execute('CREATE TRIGGER cell_changed AFTER UPDATE ON cell '
                           'BEGIN INSERT INTO _spacr_trigger_events VALUES (NEW.value+1000); END')
        connection.execute('UPDATE cell SET value=101')
        assert connection.execute('SELECT * FROM _spacr_trigger_events').fetchall() == [(42,), (1101,)]
        assert connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE name='cell_changed__cell_1'").fetchone() == ('_spacr_trigger_events',)


def test_current_filename_manifest_never_rewrites_archived_population_provenance(tmp_path):
    from tests.test_measure_original_filenames import FIELDS, _write_manifest, _write_plate

    merged = _write_plate(tmp_path, fields=FIELDS[:1])
    _write_manifest(tmp_path)
    (tmp_path / 'measurements').mkdir()
    database = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(database) as connection:
        for table in ('cell', 'cell_1'):
            connection.execute(f'CREATE TABLE {_quote(table)} '
                               '(prcf TEXT, original_filename TEXT, original_path TEXT, original_plate TEXT)')
            connection.execute(f'INSERT INTO {_quote(table)} VALUES (?, ?, ?, ?)',
                               ('plate1_r1_c1_f1', 'previous.tif', '/previous/previous.tif', 'previous'))
        connection.execute('CREATE TABLE cell_2 (prcf TEXT, area REAL)')
        connection.execute('INSERT INTO cell_2 VALUES (?, ?)', ('plate1_r1_c1_f1', 12.0))
    before_schema, before = _snapshot(database)
    result = measure._add_original_filename_columns(str(database), str(merged), report=lambda _line: None)
    after_schema, after = _snapshot(database)
    assert result == {'cell': 1}
    for table in ('cell_1', 'cell_2'):
        assert after[table] == before[table]
        assert [row for row in after_schema if row[0] == table] == [row for row in before_schema if row[0] == table]
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT original_filename, original_path FROM cell').fetchone() == (
            'well_A1_pos3.nd2', '/raw/exp7/well_A1_pos3.nd2')


@pytest.mark.parametrize('backend', ['duckdb', 'parquet', 'postgres'])
def test_non_sqlite_rerun_refuses_before_archiving_or_mixing_existing_output(tmp_path, backend):
    merged, database = _project(tmp_path)
    _seed(database)
    before = _snapshot(database)
    with pytest.raises(ConfigurationError, match='separate project folder'):
        measure._archive_measurement_run({'src': str(merged), 'measurement_backend': backend})
    assert _snapshot(database) == before


@pytest.mark.parametrize('state', ['empty', 'existing', 'busy'])
def test_windows_lock_adapter_owns_and_closes_only_its_actual_file_handle(tmp_path, monkeypatch, state):
    """Exercise the protocol with real file descriptors, not a native Windows claim."""
    import os
    import sys
    from types import SimpleNamespace

    merged, database = _project(tmp_path)
    source = (merged / 'plate1_A01_F001.npy').read_bytes()
    lock = tmp_path / '.spacr-measure-run.lock'
    if state == 'existing':
        lock.write_bytes(b'prior lock marker')
    calls = []

    class WindowsOS:
        name = 'nt'

        def __getattr__(self, name):
            return getattr(os, name)

    def locking(descriptor, operation, length):
        assert length == 1 and os.fstat(descriptor).st_size > 0
        calls.append((descriptor, operation))
        if state == 'busy' and operation == 1:
            raise OSError('lock already owned')

    with monkeypatch.context() as patch:
        patch.setattr(measure, 'os', WindowsOS())
        patch.setitem(sys.modules, 'msvcrt', SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2))
        if state == 'busy':
            with pytest.raises(ConfigurationError, match='Another Measure run'):
                with measure._measurement_run_locks([str(merged)]):
                    pytest.fail('an unavailable lock cannot admit a writer')
        else:
            with measure._measurement_run_locks([str(merged)]):
                assert calls[0][1] == 1
                assert lock.read_bytes() == (b'prior lock marker' if state == 'existing' else b'0')
    assert [operation for descriptor, operation in calls] == ([1] if state == 'busy' else [1, 2])
    assert len({descriptor for descriptor, operation in calls}) == 1
    for descriptor, operation in calls:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert (merged / 'plate1_A01_F001.npy').read_bytes() == source


def test_archive_without_an_annotation_receipt_preserves_every_population_row(tmp_path):
    merged, database = _project(tmp_path)
    _seed(database, tables=('cell',))
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE _spacr_condition_annotations (table_name TEXT, payload_json TEXT)')
    before_rows = _snapshot(database)[1]['cell']
    measure._archive_measurement_run({'src': [str(merged)]})
    assert _snapshot(database)[1]['cell_1'] == before_rows
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT * FROM _spacr_condition_annotations').fetchall() == []
