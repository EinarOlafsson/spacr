"""Tests for spacr.resume — resume / checkpointing for mask and measure.

The point of these tests is not that skipping works. Skipping is easy.
The point is that the two ways a resume can silently corrupt a dataset
are both closed:

* accepting a ``merged/*.npy`` that a crash left half-written, and
* re-measuring a field that already wrote rows, which *appends* rather
  than replaces and doubles every object in it.

``test_rerunning_without_clear_doubles_rows`` deliberately asserts the
second bug still exists when the fix is not applied, so the fix is
demonstrably load-bearing rather than decorative.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr.resume import (
    COSMETIC_SETTINGS, FIELD_KEY_COLUMNS, NON_FIELD_TABLES, REASON_EMPTY,
    REASON_PARTIAL_DB, REASON_TOO_FEW_PLANES, REASON_TRUNCATED,
    ResumeRefused, ResumeState, check_settings_compatible, clear_field_rows,
    compare_settings, completed_fields_in_db, completed_fields_in_merged,
    discover_field_tables, expected_min_planes, field_identity,
    format_resume, plan_measure_resume, plan_resume, read_npy_header,
    read_recorded_settings, resume_enabled, run_already_complete,
    validate_merged_field,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

#: Column layout of the object tables, matching what
#: ``spacr.utils._merge_and_save_to_database`` actually writes.
OBJECT_COLUMNS = ['object_label', 'plateID', 'rowID', 'columnID', 'fieldID',
                  'prcf', 'file_name', 'path_name', 'area']


def write_field(folder, stem, shape=(8, 8, 5), dtype=np.uint16):
    """Write one complete merged-style .npy field and return its path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stem + '.npy')
    np.save(path, np.zeros(shape, dtype=dtype))
    return path


def truncate(path, keep_fraction=0.5):
    """Chop a file down, exactly as a crash mid-``np.save`` would."""
    size = os.path.getsize(path)
    with open(path, 'r+b') as handle:
        handle.truncate(max(1, int(size * keep_fraction)))
    return path


def make_db(path, tables=('cell', 'nucleus', 'png_list')):
    """Create a measurements.db-shaped database with the given tables."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        for table in tables:
            columns = ', '.join(f'"{c}" TEXT' for c in OBJECT_COLUMNS)
            conn.execute(f'CREATE TABLE "{table}" ({columns})')
        conn.commit()
    finally:
        conn.close()
    return path


def measure_field(db_path, stem, n_objects=5, tables=('cell', 'nucleus',
                                                      'png_list')):
    """Append one field's rows exactly the way measure does.

    ``_merge_and_save_to_database`` and ``filepaths_to_database`` both use
    ``DataFrame.to_sql(..., if_exists='append')``. This helper reproduces
    that append faithfully — it is the behaviour the delete-before-insert
    has to compensate for.
    """
    identity = field_identity(stem)
    rows = []
    for i in range(n_objects):
        row = dict(identity)
        row['object_label'] = str(i + 1)
        row['prcf'] = '_'.join([identity['plateID'], identity['rowID'],
                                identity['columnID'], identity['fieldID']])
        row['file_name'] = stem
        row['path_name'] = stem + '.npy'
        row['area'] = str(100 + i)
        rows.append(row)
    frame = pd.DataFrame(rows)[OBJECT_COLUMNS]
    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            frame.to_sql(table, conn, if_exists='append', index=False)
        conn.commit()
    finally:
        conn.close()


def count_rows(db_path, table):
    """Row count of one table."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        conn.close()


def write_settings_table(db_path, settings):
    """Write a ``settings`` table the way ``io._save_settings_to_db`` does."""
    frame = pd.DataFrame(list(settings.items()),
                         columns=['setting_key', 'setting_value'])
    frame['setting_value'] = frame['setting_value'].apply(str)
    conn = sqlite3.connect(db_path)
    try:
        frame.to_sql('settings', conn, if_exists='replace', index=False)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def merged(tmp_path):
    """A merged/ folder with five complete fields."""
    folder = tmp_path / 'exp' / 'merged'
    for i in range(1, 6):
        write_field(str(folder), f'plate1_A01_{i}')
    return folder


# ---------------------------------------------------------------------------
# Field identity — the delete key
# ---------------------------------------------------------------------------

class TestFieldIdentity:

    def test_parses_plate_well_field(self):
        assert field_identity('plate1_A01_3') == {
            'plateID': 'plate1', 'rowID': 'r1', 'columnID': 'c1',
            'fieldID': 'f3'}

    def test_extension_and_directory_are_ignored(self):
        assert (field_identity('/data/exp/merged/plate1_B12_7.npy')
                == field_identity('plate1_B12_7'))

    def test_f1_and_f10_are_different_identities(self):
        # The whole reason deletes are keyed on columns and not on a name
        # prefix: 'f1' is a prefix of 'f10'..'f19' but not equal to any.
        assert field_identity('plate1_A01_1')['fieldID'] == 'f1'
        assert field_identity('plate1_A01_10')['fieldID'] == 'f10'
        assert (field_identity('plate1_A01_1')
                != field_identity('plate1_A01_10'))

    def test_timelapse_adds_timepoint(self):
        identity = field_identity('plate1_A01_3_2', timelapse=True)
        assert identity['timeID'] == 't2'

    def test_numeric_well_falls_back_like_map_wells(self):
        identity = field_identity('plate1_11_3')
        assert identity['rowID'] == '11' and identity['columnID'] == '11'

    def test_too_few_parts_refuses_rather_than_guessing(self):
        # A guessed key is a key that deletes the wrong rows.
        with pytest.raises(ValueError, match='cannot identify a field'):
            field_identity('junk')

    def test_mapping_missing_keys_refuses(self):
        with pytest.raises(ValueError, match='missing'):
            field_identity({'plateID': 'plate1', 'rowID': 'r1'})

    def test_mapping_passthrough(self):
        identity = {'plateID': 'p', 'rowID': 'r1', 'columnID': 'c1',
                    'fieldID': 'f9'}
        assert field_identity(identity) == identity

    def test_timelapse_without_time_component_refuses(self):
        with pytest.raises(ValueError, match='time component'):
            field_identity('plate1_A01_3', timelapse=True)

    def test_agrees_with_utils_map_wells(self):
        """The local copy must not drift from spacr.utils._map_wells.

        resume.py cannot import spacr.utils (it drags in torch and
        cellpose, and resume runs before any model is loaded), so the
        parsing is duplicated. This pins the duplicate to the original.
        """
        from spacr.utils import _map_wells
        for name in ['plate1_A01_3', 'plate1_B12_10', 'plate1_H09_1',
                     'plate2_A01_19', 'plate1_11_3']:
            plate, row, column, fld, _prcf = _map_wells(name)
            assert field_identity(name) == {
                'plateID': plate, 'rowID': row, 'columnID': column,
                'fieldID': fld}, name

    def test_agrees_with_utils_map_wells_timelapse(self):
        from spacr.utils import _map_wells
        name = 'plate1_A01_3_7'
        plate, row, column, fld, timeid, _prcf = _map_wells(name,
                                                            timelapse=True)
        assert field_identity(name, timelapse=True) == {
            'plateID': plate, 'rowID': row, 'columnID': column,
            'fieldID': fld, 'timeID': timeid}


# ---------------------------------------------------------------------------
# .npy validation — "it exists" is not "it finished"
# ---------------------------------------------------------------------------

class TestValidation:

    def test_complete_field_validates(self, tmp_path):
        path = write_field(str(tmp_path), 'plate1_A01_1')
        assert validate_merged_field(path) == (True, 'done')

    def test_header_reports_real_shape(self, tmp_path):
        path = write_field(str(tmp_path), 'plate1_A01_1', shape=(4, 6, 7))
        header = read_npy_header(path)
        assert header['shape'] == (4, 6, 7)
        assert header['itemsize'] == 2
        assert header['actual_bytes'] == header['expected_bytes']

    def test_truncated_npy_is_pending_not_done(self, tmp_path):
        """THE test. A crash mid-``np.save`` leaves a short file at the
        final name; ``np.save`` writes in place, so this is the state a
        killed mask run really leaves behind. A resume that trusted
        ``os.path.exists`` would skip it and every downstream number
        computed from that field would be garbage that looks like data.
        """
        folder = tmp_path / 'merged'
        write_field(str(folder), 'plate1_A01_1')
        write_field(str(folder), 'plate1_A01_2')
        bad = write_field(str(folder), 'plate1_A01_3')
        truncate(bad, keep_fraction=0.4)

        # It still exists, and it still looks like a .npy.
        assert os.path.isfile(bad)
        assert open(bad, 'rb').read(6) == b'\x93NUMPY'

        assert validate_merged_field(bad) == (False, REASON_TRUNCATED)

        reasons = {}
        done = completed_fields_in_merged(str(folder), reasons=reasons)
        assert done == {'plate1_A01_1', 'plate1_A01_2'}
        assert reasons == {'plate1_A01_3': REASON_TRUNCATED}

        state = plan_resume(['plate1_A01_1', 'plate1_A01_2', 'plate1_A01_3'],
                            done, reasons=reasons)
        assert state.pending == ('plate1_A01_3',)
        assert state.n_rejected == 1
        assert state.rejection_counts() == {REASON_TRUNCATED: 1}
        assert REASON_TRUNCATED in format_resume(state)

    def test_numpy_also_refuses_to_load_the_truncated_file(self, tmp_path):
        """Cross-check: the header check agrees with numpy, but is cheaper.

        ``np.load`` has to allocate the whole array before it discovers
        the file is short; the header check reads 100 bytes.
        """
        path = truncate(write_field(str(tmp_path), 'plate1_A01_1',
                                    shape=(64, 64, 5)), keep_fraction=0.3)
        with pytest.raises(Exception):
            np.load(path)
        assert validate_merged_field(path)[1] == REASON_TRUNCATED

    def test_zero_byte_npy_is_pending_and_counted_as_rejected(self, tmp_path):
        folder = tmp_path / 'merged'
        write_field(str(folder), 'plate1_A01_1')
        empty = os.path.join(str(folder), 'plate1_A01_2.npy')
        open(empty, 'wb').close()
        assert os.path.getsize(empty) == 0

        reasons = {}
        done = completed_fields_in_merged(str(folder), reasons=reasons)
        assert done == {'plate1_A01_1'}
        assert reasons == {'plate1_A01_2': REASON_EMPTY}

        state = plan_resume(['plate1_A01_1', 'plate1_A01_2'], done,
                            reasons=reasons)
        assert state.n_rejected == 1
        assert 'plate1_A01_2' in state.rejected

    def test_garbage_file_is_rejected(self, tmp_path):
        path = os.path.join(str(tmp_path), 'plate1_A01_1.npy')
        Path(path).write_bytes(b'not an npy at all, just bytes')
        ok, reason = validate_merged_field(path)
        assert ok is False and reason == 'unreadable'

    def test_missing_file_is_missing(self, tmp_path):
        ok, reason = validate_merged_field(str(tmp_path / 'nope.npy'))
        assert (ok, reason) == (False, 'missing')

    def test_too_few_planes_rejected_against_explicit_floor(self, tmp_path):
        thin = write_field(str(tmp_path), 'plate1_A01_1', shape=(8, 8, 3))
        assert validate_merged_field(thin, min_planes=5) == (
            False, REASON_TOO_FEW_PLANES)
        assert validate_merged_field(thin, min_planes=3)[0] is True

    def test_modal_plane_count_catches_the_odd_one_out(self, tmp_path):
        """Every field in a merged folder is written by one loop with one
        channel layout, so a field with fewer planes than its neighbours
        did not finish."""
        folder = tmp_path / 'merged'
        for i in range(1, 5):
            write_field(str(folder), f'plate1_A01_{i}', shape=(8, 8, 6))
        write_field(str(folder), 'plate1_A01_5', shape=(8, 8, 3))
        reasons = {}
        done = completed_fields_in_merged(str(folder), reasons=reasons)
        assert 'plate1_A01_5' not in done
        assert reasons['plate1_A01_5'] == REASON_TOO_FEW_PLANES

    def test_missing_folder_yields_nothing(self, tmp_path):
        assert completed_fields_in_merged(str(tmp_path / 'nope')) == set()

    def test_read_npy_header_rejects_empty_file(self, tmp_path):
        path = tmp_path / 'x.npy'
        path.write_bytes(b'')
        with pytest.raises(ValueError, match='zero bytes'):
            read_npy_header(str(path))

    def test_expected_min_planes_from_settings(self):
        assert expected_min_planes({'channels': [0, 1, 2],
                                    'cell_mask_dim': 4,
                                    'nucleus_mask_dim': 5}) == 6
        assert expected_min_planes({'channels': [], 'cell_mask_dim': None}) is None
        assert expected_min_planes('not a mapping') is None


class TestPartiallyPopulatedMerged:

    def test_pending_set_is_exactly_what_is_missing(self, tmp_path):
        folder = tmp_path / 'merged'
        all_fields = [f'plate1_A01_{i}' for i in range(1, 11)]
        for stem in all_fields[:6]:          # a run that died at 6 of 10
            write_field(str(folder), stem)

        done = completed_fields_in_merged(str(folder))
        state = plan_resume(all_fields, done, src=str(folder))

        assert state.total == 10
        assert state.n_skipped == 6
        assert state.pending == tuple(all_fields[6:])
        assert state.n_rejected == 0        # nothing was present-but-bad
        assert set(state.done) == set(all_fields[:6])

    def test_filter_files_preserves_order_and_extension(self, tmp_path):
        folder = tmp_path / 'merged'
        for stem in ['plate1_A01_1', 'plate1_A01_2']:
            write_field(str(folder), stem)
        files = ['plate1_A01_1.npy', 'plate1_A01_2.npy', 'plate1_A01_3.npy']
        state = plan_resume([f[:-4] for f in files],
                            completed_fields_in_merged(str(folder)))
        assert state.filter_files(files) == ['plate1_A01_3.npy']


# ---------------------------------------------------------------------------
# Table discovery
# ---------------------------------------------------------------------------

class TestDiscoverFieldTables:

    def test_finds_every_per_field_table(self, tmp_path):
        db = make_db(str(tmp_path / 'measurements' / 'measurements.db'),
                     tables=('cell', 'nucleus', 'pathogen', 'cytoplasm',
                             'organelle', 'cell_organelle_summary',
                             'png_list'))
        assert set(discover_field_tables(db)) == {
            'cell', 'nucleus', 'pathogen', 'cytoplasm', 'organelle',
            'cell_organelle_summary', 'png_list'}

    def test_excludes_tables_owned_by_other_stages(self, tmp_path):
        """object_counts / pivoted_counts belong to the mask stage and
        settings / run_status are run metadata. None of them may be
        cleared by a measure resume — and none of them carries the four
        well columns, so the column test alone already excludes them."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'),
                     tables=('cell',))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE object_counts (file_name TEXT, '
                     'count_type TEXT, object_count INTEGER)')
        conn.execute('CREATE TABLE pivoted_counts (file_name TEXT, cell REAL)')
        conn.execute('CREATE TABLE settings (setting_key TEXT, '
                     'setting_value TEXT)')
        conn.commit()
        conn.close()
        assert discover_field_tables(db) == ['cell']

    def test_table_with_only_some_key_columns_is_not_a_field_table(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'), tables=('cell',))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE half (plateID TEXT, rowID TEXT)')
        conn.commit()
        conn.close()
        assert 'half' not in discover_field_tables(db)

    def test_missing_database_yields_nothing(self, tmp_path):
        assert discover_field_tables(str(tmp_path / 'nope.db')) == []

    def test_non_field_tables_constant_covers_the_known_names(self):
        assert {'object_counts', 'pivoted_counts', 'settings',
                'run_status'} <= NON_FIELD_TABLES


# ---------------------------------------------------------------------------
# completed_fields_in_db
# ---------------------------------------------------------------------------

class TestCompletedFieldsInDb:

    def test_field_in_every_table_is_done(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_1')
        measure_field(db, 'plate1_A01_2')
        done = completed_fields_in_db(db, fields=['plate1_A01_1',
                                                  'plate1_A01_2',
                                                  'plate1_A01_3'])
        assert done == {'plate1_A01_1', 'plate1_A01_2'}

    def test_field_in_some_tables_is_partial_not_done(self, tmp_path):
        """A worker killed between the cell insert and the nucleus insert
        leaves a field that has cells and no nuclei. Calling that done
        loses its nuclei forever; re-running it without clearing doubles
        its cells. It must come back as partial so the caller does the
        third, correct thing."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_1')
        measure_field(db, 'plate1_A01_2', tables=('cell',))  # crash here

        partial = {}
        done = completed_fields_in_db(db, fields=['plate1_A01_1',
                                                  'plate1_A01_2'],
                                      partial=partial)
        assert done == {'plate1_A01_1'}
        assert partial == {'plate1_A01_2': REASON_PARTIAL_DB}

    def test_require_all_false_accepts_any_table(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_2', tables=('cell',))
        done = completed_fields_in_db(db, fields=['plate1_A01_2'],
                                      require_all=False)
        assert done == {'plate1_A01_2'}

    def test_without_candidates_returns_prcf_labels(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_4')
        assert completed_fields_in_db(db) == {'plate1_r1_c1_f4'}

    def test_missing_database_yields_nothing(self, tmp_path):
        assert completed_fields_in_db(str(tmp_path / 'nope.db')) == set()

    def test_unparseable_candidate_is_ignored(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_1')
        assert completed_fields_in_db(db, fields=['junk', 'plate1_A01_1']) == {
            'plate1_A01_1'}


# ---------------------------------------------------------------------------
# clear_field_rows — the delete-before-insert
# ---------------------------------------------------------------------------

class TestClearFieldRows:

    def test_deletes_from_every_table_the_field_touched(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'),
                     tables=('cell', 'nucleus', 'pathogen', 'cytoplasm',
                             'organelle', 'cell_organelle_summary',
                             'png_list'))
        tables = ('cell', 'nucleus', 'pathogen', 'cytoplasm', 'organelle',
                  'cell_organelle_summary', 'png_list')
        measure_field(db, 'plate1_A01_1', n_objects=3, tables=tables)
        measure_field(db, 'plate1_A01_2', n_objects=3, tables=tables)

        deleted = clear_field_rows(db, None, 'plate1_A01_1')

        assert deleted == 3 * len(tables)
        for table in tables:
            assert count_rows(db, table) == 3      # only field 2 remains
            conn = sqlite3.connect(db)
            remaining = conn.execute(
                f'SELECT DISTINCT fieldID FROM "{table}"').fetchall()
            conn.close()
            assert remaining == [('f2',)]

    def test_deleting_f1_does_not_touch_f10_through_f19(self, tmp_path):
        """A LIKE-based delete on 'plate1_r1_c1_f1%' matches nineteen
        innocent fields. Equality on fieldID matches one."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        fields = ['plate1_A01_1'] + [f'plate1_A01_{i}' for i in range(10, 20)]
        for stem in fields:
            measure_field(db, stem, n_objects=2)
        assert count_rows(db, 'cell') == len(fields) * 2

        deleted = clear_field_rows(db, None, 'plate1_A01_1')

        assert deleted == 2 * 3                     # 2 rows x 3 tables
        assert count_rows(db, 'cell') == 10 * 2     # f10..f19 untouched
        conn = sqlite3.connect(db)
        survivors = {r[0] for r in
                     conn.execute('SELECT DISTINCT fieldID FROM cell')}
        conn.close()
        assert survivors == {f'f{i}' for i in range(10, 20)}

    def test_the_like_trap_is_real(self, tmp_path):
        """Documents why the delete is keyed on columns, not names: the
        obvious prefix match really does destroy f10-f19."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'),
                     tables=('cell',))
        for stem in ['plate1_A01_1'] + [f'plate1_A01_{i}'
                                        for i in range(10, 20)]:
            measure_field(db, stem, n_objects=2, tables=('cell',))
        conn = sqlite3.connect(db)
        cursor = conn.execute(
            "DELETE FROM cell WHERE prcf LIKE 'plate1_r1_c1_f1%'")
        naive = cursor.rowcount
        conn.commit()
        conn.close()
        assert naive == 22          # all eleven fields, not one
        assert count_rows(db, 'cell') == 0

    def test_rolls_back_when_a_delete_fails_part_way(self, tmp_path):
        """All or nothing. A half-cleared field is worse than an
        uncleared one: its cells would be gone and its nuclei doubled."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'),
                     tables=('cell', 'nucleus'))
        measure_field(db, 'plate1_A01_1', n_objects=4,
                      tables=('cell', 'nucleus'))
        measure_field(db, 'plate1_A01_2', n_objects=4,
                      tables=('cell', 'nucleus'))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TRIGGER boom BEFORE DELETE ON nucleus '
                     "BEGIN SELECT RAISE(ABORT, 'boom'); END")
        conn.commit()
        conn.close()

        before = (count_rows(db, 'cell'), count_rows(db, 'nucleus'))
        with pytest.raises(sqlite3.Error):
            clear_field_rows(db, ['cell', 'nucleus'], 'plate1_A01_1')
        assert (count_rows(db, 'cell'), count_rows(db, 'nucleus')) == before

    def test_refuses_a_table_without_the_key_columns(self, tmp_path):
        """Pre-flight: a table keyed on only plateID would have every
        field in the plate deleted. Refuse before touching anything."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'),
                     tables=('cell',))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE half (plateID TEXT, rowID TEXT)')
        conn.execute("INSERT INTO half VALUES ('plate1', 'r1')")
        conn.commit()
        conn.close()
        measure_field(db, 'plate1_A01_1', n_objects=3, tables=('cell',))

        with pytest.raises(ValueError, match='Refusing'):
            clear_field_rows(db, ['cell', 'half'], 'plate1_A01_1')
        assert count_rows(db, 'cell') == 3      # nothing was deleted
        assert count_rows(db, 'half') == 1

    def test_refuses_a_table_that_does_not_exist(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'), tables=('cell',))
        measure_field(db, 'plate1_A01_1', tables=('cell',))
        with pytest.raises(ValueError, match='does not exist'):
            clear_field_rows(db, ['cell', 'ghost'], 'plate1_A01_1')
        assert count_rows(db, 'cell') == 5

    def test_missing_database_is_a_no_op(self, tmp_path):
        assert clear_field_rows(str(tmp_path / 'nope.db'), None,
                                'plate1_A01_1') == 0

    def test_no_field_tables_is_a_no_op(self, tmp_path):
        db = str(tmp_path / 'm' / 'measurements.db')
        os.makedirs(os.path.dirname(db))
        sqlite3.connect(db).close()
        assert clear_field_rows(db, None, 'plate1_A01_1') == 0

    def test_non_field_tables_are_filtered_out_even_if_named(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'), tables=('cell',))
        measure_field(db, 'plate1_A01_1', n_objects=2, tables=('cell',))
        # 'settings' is on the deny-list, so naming it changes nothing.
        assert clear_field_rows(db, ['cell', 'settings'], 'plate1_A01_1') == 2

    def test_timelapse_clears_one_frame_only(self, tmp_path):
        db = str(tmp_path / 'm' / 'measurements.db')
        os.makedirs(os.path.dirname(db))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, '
                     'columnID TEXT, fieldID TEXT, timeID TEXT)')
        for t in range(1, 4):
            conn.execute('INSERT INTO cell VALUES (?,?,?,?,?)',
                         ('plate1', 'r1', 'c1', 'f3', f't{t}'))
        conn.commit()
        conn.close()

        assert clear_field_rows(db, ['cell'], 'plate1_A01_3_2',
                                timelapse=True) == 1
        assert count_rows(db, 'cell') == 2


# ---------------------------------------------------------------------------
# Idempotency — the actual deliverable
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_rerunning_without_clear_doubles_rows(self, tmp_path):
        """THE BUG THE FIX EXISTS FOR — asserted, so the fix is provably
        load-bearing.

        ``_merge_and_save_to_database`` uses
        ``to_sql(if_exists='append')``. Measuring a field twice therefore
        stores every object twice, every per-well count is inflated, and
        nothing anywhere records that it happened.
        """
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_3', n_objects=7)
        first = count_rows(db, 'cell')

        measure_field(db, 'plate1_A01_3', n_objects=7)   # a naive resume

        assert count_rows(db, 'cell') == 2 * first
        assert first == 7

    def test_clear_then_rerun_gives_the_same_row_count(self, tmp_path):
        """The fix: delete-before-insert makes re-measuring idempotent."""
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_3', n_objects=7)
        baseline = {t: count_rows(db, t) for t in ('cell', 'nucleus',
                                                   'png_list')}

        clear_field_rows(db, None, 'plate1_A01_3')
        measure_field(db, 'plate1_A01_3', n_objects=7)

        assert {t: count_rows(db, t) for t in baseline} == baseline

    def test_clear_then_rerun_leaves_other_fields_alone(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        for i in range(1, 4):
            measure_field(db, f'plate1_A01_{i}', n_objects=4)
        baseline = count_rows(db, 'cell')

        clear_field_rows(db, None, 'plate1_A01_2')
        measure_field(db, 'plate1_A01_2', n_objects=4)

        assert count_rows(db, 'cell') == baseline

    def test_repeated_clear_and_rerun_never_grows(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        measure_field(db, 'plate1_A01_3', n_objects=5)
        baseline = count_rows(db, 'cell')
        for _ in range(4):
            clear_field_rows(db, None, 'plate1_A01_3')
            measure_field(db, 'plate1_A01_3', n_objects=5)
            assert count_rows(db, 'cell') == baseline


# ---------------------------------------------------------------------------
# Settings compatibility
# ---------------------------------------------------------------------------

class TestSettingsCompatibility:

    def test_material_change_blocks_and_names_what_differs(self):
        recorded = {'cell_mask_dim': 4, 'channels': [0, 1, 2],
                    'cell_min_size': 100}
        current = {'cell_mask_dim': 5, 'channels': [0, 1, 2],
                   'cell_min_size': 100}
        with pytest.raises(ResumeRefused) as excinfo:
            check_settings_compatible(recorded, current, source='db')
        message = str(excinfo.value)
        assert 'cell_mask_dim' in message
        assert '4' in message and '5' in message
        assert 'cell_min_size' not in message      # unchanged, not named

    def test_channel_and_diameter_and_crop_changes_all_block(self):
        for key, old, new in [('channels', [0, 1, 2], [0, 1]),
                              ('cell_diameter', 60, 30),
                              ('png_size', [[128, 128]], [[224, 224]]),
                              ('crop_mode', ['cell'], ['nucleus']),
                              ('experiment', 'exp1', 'exp2')]:
            with pytest.raises(ResumeRefused, match=key):
                check_settings_compatible({key: old}, {key: new})

    def test_env_drift_alone_does_not_block(self):
        recorded = {'cell_mask_dim': 4, 'numpy': '1.26.0', 'torch': '2.1.0'}
        current = {'cell_mask_dim': 4, 'numpy': '2.0.1', 'torch': '2.4.0'}
        comparison = check_settings_compatible(recorded, current)
        assert comparison.blocks_resume is False
        assert {e['key'] for e in comparison.env} == {'numpy', 'torch'}
        assert comparison.changed == ()

    def test_cosmetic_drift_alone_does_not_block(self):
        recorded = {'cell_mask_dim': 4, 'n_jobs': 8, 'plot': True,
                    'verbose': False, 'src': '/a/merged'}
        current = {'cell_mask_dim': 4, 'n_jobs': 2, 'plot': False,
                   'verbose': True, 'src': '/b/merged'}
        comparison = check_settings_compatible(recorded, current)
        assert comparison.blocks_resume is False
        assert {e['key'] for e in comparison.drift} == {'n_jobs', 'plot',
                                                        'verbose', 'src'}

    def test_csv_string_round_trip_is_not_a_change(self):
        """Settings reach the db stringified by ``_save_settings_to_db``.
        '[0, 1, 2]' and [0, 1, 2] are the same setting recorded twice."""
        comparison = compare_settings({'channels': '[0, 1, 2]',
                                       'cell_mask_dim': '4',
                                       'timelapse': 'False',
                                       'organelle_channel': 'None'},
                                      {'channels': [0, 1, 2],
                                       'cell_mask_dim': 4,
                                       'timelapse': False,
                                       'organelle_channel': None})
        assert comparison.changed == ()
        assert comparison.same == 4

    def test_new_material_key_with_a_real_value_blocks(self):
        """Turning on an organelle channel the old run never had really
        does change what is measured, even though the key is new."""
        with pytest.raises(ResumeRefused, match='organelle_channel'):
            check_settings_compatible({'cell_mask_dim': 4},
                                      {'cell_mask_dim': 4,
                                       'organelle_channel': 3})

    def test_new_key_with_an_inert_value_does_not_block(self):
        comparison = check_settings_compatible(
            {'cell_mask_dim': 4},
            {'cell_mask_dim': 4, 'organelle_channel': None, 'n_jobs': 4})
        assert comparison.blocks_resume is False
        assert set(comparison.only_in_current) == {'organelle_channel',
                                                   'n_jobs'}

    def test_dropped_key_is_schema_drift_not_a_change(self):
        comparison = check_settings_compatible(
            {'cell_mask_dim': 4, 'legacy_option': 7}, {'cell_mask_dim': 4})
        assert comparison.blocks_resume is False
        assert comparison.only_in_recorded == ('legacy_option',)

    def test_describe_lists_every_material_change(self):
        comparison = compare_settings({'a': 1, 'b': 2}, {'a': 9, 'b': 8})
        described = comparison.describe()
        assert 'a:' in described and 'b:' in described

    def test_cosmetic_set_contains_the_obvious_knobs(self):
        assert {'n_jobs', 'plot', 'verbose', 'src', 'resume'} <= COSMETIC_SETTINGS


class TestReadRecordedSettings:

    def test_reads_the_settings_table_measure_writes(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'), tables=('cell',))
        write_settings_table(db, {'cell_mask_dim': 4, 'channels': [0, 1, 2]})
        recorded = read_recorded_settings(db)
        assert recorded['cell_mask_dim'] == '4'
        assert recorded['channels'] == '[0, 1, 2]'

    def test_database_without_a_settings_table_reads_as_no_information(self, tmp_path):
        db = make_db(str(tmp_path / 'm' / 'measurements.db'), tables=('cell',))
        assert read_recorded_settings(db) == {}

    def test_reads_a_key_value_csv(self, tmp_path):
        path = tmp_path / 'gen_mask_settings.csv'
        path.write_text('Key,Value\ncell_channel,0\nnucleus_channel,1\n')
        assert read_recorded_settings(str(path)) == {'cell_channel': '0',
                                                     'nucleus_channel': '1'}

    def test_reads_a_settings_json(self, tmp_path):
        path = tmp_path / 'settings.json'
        path.write_text('{"cell_mask_dim": 4}')
        assert read_recorded_settings(str(path)) == {'cell_mask_dim': 4}

    def test_missing_and_empty_sources_read_as_empty(self, tmp_path):
        assert read_recorded_settings('') == {}
        assert read_recorded_settings(str(tmp_path / 'nope.db')) == {}
        assert read_recorded_settings(str(tmp_path / 'nope.txt')) == {}
        other = tmp_path / 'x.txt'
        other.write_text('hi')
        assert read_recorded_settings(str(other)) == {}


# ---------------------------------------------------------------------------
# run_status
# ---------------------------------------------------------------------------

class TestRunStatus:

    def test_a_run_stamped_complete_needs_no_resume(self, tmp_path):
        from spacr.errors import RunLedger
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        ledger = RunLedger('measure_crop')
        for i in range(1, 4):
            ledger.record_success(f'plate1_A01_{i}.npy', stage='measure')
        ledger.stamp(db)

        assert run_already_complete(db) is True
        assert run_already_complete(db, name='measure_crop') is True
        assert run_already_complete(db, name='preprocess_generate_masks') is False

    def test_a_partial_run_is_not_complete(self, tmp_path):
        from spacr.errors import RunLedger
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        ledger = RunLedger('measure_crop')
        ledger.record_success('plate1_A01_1.npy', stage='measure')
        ledger.record_failure('plate1_A01_2.npy', stage='measure',
                              exc='field produced no result')
        ledger.stamp(db)
        assert run_already_complete(db, name='measure_crop') is False

    def test_an_unstamped_artifact_is_not_complete(self, tmp_path):
        """Stricter than errors.run_is_complete on purpose: for a resume,
        "nobody ever said" must mean "go and check the files"."""
        from spacr.errors import run_is_complete
        db = make_db(str(tmp_path / 'm' / 'measurements.db'))
        assert run_is_complete(db) is True        # the errors.py semantics
        assert run_already_complete(db) is False  # the resume semantics

    def test_missing_database_is_not_complete(self, tmp_path):
        assert run_already_complete(str(tmp_path / 'nope.db')) is False

    def test_plan_reports_nothing_to_do_when_everything_is_measured(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        for i in range(1, 4):
            write_field(str(src), f'plate1_A01_{i}')
            measure_field(db, f'plate1_A01_{i}')

        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)
        assert state.n_pending == 0
        assert state.n_skipped == 3
        assert 'Nothing to do' in format_resume(state)


# ---------------------------------------------------------------------------
# plan_measure_resume — the end-to-end entry point measure.py calls
# ---------------------------------------------------------------------------

class TestPlanMeasureResume:

    def test_returns_none_when_resume_is_not_requested(self, tmp_path):
        """Resume is opt-in: without the flag the plan is None and the
        caller's file list is untouched."""
        src = tmp_path / 'exp' / 'merged'
        write_field(str(src), 'plate1_A01_1')
        assert plan_measure_resume({'src': str(src)}) is None
        assert plan_measure_resume({'src': str(src), 'resume': False}) is None
        assert plan_measure_resume({}) is None

    def test_resume_enabled_accepts_strings_and_rejects_junk(self):
        assert resume_enabled({'resume': True}) is True
        assert resume_enabled({'resume': 'yes'}) is True
        assert resume_enabled({'resume': 'no'}) is False
        assert resume_enabled({}) is False
        assert resume_enabled(None) is False

    def test_disabled_state_changes_nothing(self):
        files = ['a_A01_1.npy', 'a_A01_2.npy']
        state = plan_resume([f[:-4] for f in files], ['a_A01_1'],
                            enabled=False)
        assert state.filter_files(files) == files
        assert state.pending == ('a_A01_1', 'a_A01_2')
        assert state.n_skipped == 0
        assert 'OFF' in format_resume(state)

    def test_skips_measured_fields_and_runs_the_rest(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        for i in range(1, 11):
            write_field(str(src), f'plate1_A01_{i}')
        for i in range(1, 8):                    # died at 7 of 10
            measure_field(db, f'plate1_A01_{i}')

        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)
        assert state.total == 10
        assert state.n_skipped == 7
        assert set(state.pending) == {f'plate1_A01_{i}' for i in range(8, 11)}

    def test_clears_rows_of_a_partially_written_field_before_rerun(self, tmp_path):
        """End-to-end idempotency: a field the crash left with cell rows
        and no nucleus rows is cleared, so re-measuring it produces one
        copy, not one-and-a-half."""
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        for i in (1, 2):
            write_field(str(src), f'plate1_A01_{i}')
        measure_field(db, 'plate1_A01_1', n_objects=4)
        measure_field(db, 'plate1_A01_2', n_objects=4, tables=('cell',))

        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)

        assert state.pending == ('plate1_A01_2',)
        assert state.reasons['plate1_A01_2'] == REASON_PARTIAL_DB
        assert state.cleared_rows == 4              # the orphaned cell rows
        assert count_rows(db, 'cell') == 4          # only field 1 left

        measure_field(db, 'plate1_A01_2', n_objects=4)
        assert count_rows(db, 'cell') == 8          # not 12

    def test_truncated_merged_field_is_rerun_and_cleared(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        write_field(str(src), 'plate1_A01_1')
        truncate(write_field(str(src), 'plate1_A01_2'), 0.3)
        measure_field(db, 'plate1_A01_1')
        measure_field(db, 'plate1_A01_2')     # measured from a bad stack

        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)

        assert state.pending == ('plate1_A01_2',)
        assert state.reasons['plate1_A01_2'] == REASON_TRUNCATED
        assert state.n_rejected == 1
        assert state.cleared_rows == 15       # 5 rows x 3 tables, removed
        assert count_rows(db, 'cell') == 5

    def test_settings_change_refuses_the_resume(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        write_field(str(src), 'plate1_A01_1')
        measure_field(db, 'plate1_A01_1')
        write_settings_table(db, {'src': str(src), 'cell_mask_dim': 4,
                                  'channels': [0, 1, 2]})

        with pytest.raises(ResumeRefused, match='cell_mask_dim'):
            plan_measure_resume({'src': str(src), 'resume': True,
                                 'cell_mask_dim': 5, 'channels': [0, 1, 2]},
                                verbose=False)

    def test_identical_settings_allow_the_resume(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        for i in (1, 2):
            write_field(str(src), f'plate1_A01_{i}')
        measure_field(db, 'plate1_A01_1')
        settings = {'src': str(src), 'cell_mask_dim': 4,
                    'channels': [0, 1, 2], 'n_jobs': 8}
        write_settings_table(db, settings)

        resumed = dict(settings, resume=True, n_jobs=2)  # cosmetic drift only
        state = plan_measure_resume(resumed, verbose=False)
        assert state.pending == ('plate1_A01_2',)

    def test_only_dirty_fields_are_cleared(self, tmp_path, monkeypatch):
        """A resume clears the field that was mid-flight, not all hundred.

        Issuing a DELETE for a field that has no rows costs a full scan of
        a million-row table to delete nothing, once per field.
        """
        import spacr.resume as module
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        for i in range(1, 6):
            write_field(str(src), f'plate1_A01_{i}')
        measure_field(db, 'plate1_A01_1')                      # complete
        measure_field(db, 'plate1_A01_2', tables=('cell',))    # mid-flight

        cleared_for = []
        real = module.clear_field_rows

        def spy(db_path, tables, field, timelapse=False):
            cleared_for.append(field)
            return real(db_path, tables, field, timelapse=timelapse)

        monkeypatch.setattr(module, 'clear_field_rows', spy)
        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)

        assert set(state.pending) == {f'plate1_A01_{i}' for i in range(2, 6)}
        assert cleared_for == ['plate1_A01_2']   # only the dirty one

    def test_no_database_yet_means_everything_is_pending(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        for i in (1, 2, 3):
            write_field(str(src), f'plate1_A01_{i}')
        state = plan_measure_resume({'src': str(src), 'resume': True},
                                    verbose=False)
        assert state.n_pending == 3 and state.cleared_rows == 0

    def test_min_planes_comes_from_the_settings(self, tmp_path):
        """A stack with fewer planes than cell_mask_dim needs cannot be
        measured, however complete its bytes are."""
        src = tmp_path / 'exp' / 'merged'
        write_field(str(src), 'plate1_A01_1', shape=(8, 8, 8))
        write_field(str(src), 'plate1_A01_2', shape=(8, 8, 4))
        state = plan_measure_resume(
            {'src': str(src), 'resume': True, 'channels': [0, 1, 2],
             'cell_mask_dim': 6}, verbose=False)
        assert state.reasons['plate1_A01_2'] == REASON_TOO_FEW_PLANES

    def test_verbose_prints_the_plan_before_doing_anything(self, tmp_path, capsys):
        src = tmp_path / 'exp' / 'merged'
        write_field(str(src), 'plate1_A01_1')
        plan_measure_resume({'src': str(src), 'resume': True}, verbose=True)
        out = capsys.readouterr().out
        assert 'spaCR RESUME' in out and 'to run' in out

    def test_a_field_that_cannot_be_cleared_refuses_rather_than_doubling(self, tmp_path):
        src = tmp_path / 'exp' / 'merged'
        db = make_db(str(tmp_path / 'exp' / 'measurements' /
                         'measurements.db'))
        write_field(str(src), 'plate1_A01_1')
        measure_field(db, 'plate1_A01_1', tables=('cell',))
        conn = sqlite3.connect(db)
        conn.execute('CREATE TRIGGER boom BEFORE DELETE ON cell '
                     "BEGIN SELECT RAISE(ABORT, 'boom'); END")
        conn.commit()
        conn.close()

        with pytest.raises(ResumeRefused, match='inflate'):
            plan_measure_resume({'src': str(src), 'resume': True},
                                verbose=False)

    def test_missing_merged_folder_plans_nothing(self, tmp_path):
        state = plan_measure_resume(
            {'src': str(tmp_path / 'nope' / 'merged'), 'resume': True},
            verbose=False)
        assert state.total == 0 and state.pending == ()


class TestResumeStateReport:

    def test_format_lists_counts_and_reasons(self):
        state = ResumeState(
            total=1000, done=tuple(f'f{i}' for i in range(900)),
            pending=('a', 'b', 'c'), skipped=tuple(f'f{i}' for i in range(900)),
            reasons={'a': REASON_TRUNCATED, 'b': REASON_TRUNCATED,
                     'c': 'not-measured'},
            cleared_rows=42, src='/data/merged', db_path='/data/m.db')
        text = format_resume(state)
        assert '1000' in text and '900' in text
        assert 'truncated' in text and 'x2' in text
        assert '42 stale row' in text
        assert '/data/merged' in text and '/data/m.db' in text
        assert state.n_rejected == 2

    def test_many_rejections_are_elided(self):
        names = [f'plate1_A01_{i}' for i in range(20)]
        state = plan_resume(names, [], reasons={n: REASON_TRUNCATED
                                                for n in names})
        assert '+16 more' in format_resume(state, max_examples=4)

    def test_duplicate_candidates_are_collapsed(self):
        state = plan_resume(['a_A01_1', 'a_A01_1.npy'], [])
        assert state.total == 1


# ---------------------------------------------------------------------------
# Dependency weight — resume must never drag in torch or cellpose
# ---------------------------------------------------------------------------

def test_resume_imports_no_torch_or_cellpose():
    """resume runs at the very top of measure_crop, before any model is
    loaded, and in a process that may never load one. Importing torch or
    cellpose here would cost seconds and gigabytes for a question that is
    answered by reading file headers and a sqlite table.
    """
    repo_root = Path(__file__).resolve().parent.parent
    code = (
        'import sys\n'
        'import spacr.resume\n'
        'heavy = [m for m in ("torch", "cellpose", "numpy", "pandas",\n'
        '                     "matplotlib", "skimage", "cv2")\n'
        '         if m in sys.modules]\n'
        'print(",".join(heavy))\n'
    )
    # A clean PYTHONPATH: the surrounding test run may inject a
    # sitecustomize that pre-imports torch (the coverage shim does), which
    # would make this pass or fail for reasons unrelated to resume.py.
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run([sys.executable, '-c', code], cwd=str(repo_root),
                            capture_output=True, text=True, timeout=180,
                            env=env)
    assert result.returncode == 0, result.stderr
    loaded = [m for m in result.stdout.strip().split(',') if m]
    assert loaded == [], f'spacr.resume pulled in heavy modules: {loaded}'


def test_resume_module_only_uses_light_imports():
    import spacr.resume as module
    source = Path(module.__file__).read_text()
    for banned in ('import torch', 'import cellpose', 'from torch',
                   'from cellpose', 'import numpy', 'import cv2'):
        assert banned not in source, banned


def test_mask_resume_requeues_a_truncated_array(tmp_path):
    from spacr.io import _check_masks

    output = tmp_path / "masks"
    output.mkdir()
    (output / "field1.npy").write_bytes(b"\x93NUMPY")
    batch = np.stack([
        np.zeros((4, 4), dtype=np.uint16),
        np.ones((4, 4), dtype=np.uint16),
    ])

    pending, names = _check_masks(
        batch, ["field1.npy", "field2.npy"], str(output), resume=True)

    assert names == ["field1.npy", "field2.npy"]
    assert pending.shape[0] == 2


def test_mask_folder_resume_counts_only_complete_fields(tmp_path):
    from spacr.io import _save_array_atomic
    from spacr.utils import check_mask_folder

    src = tmp_path / "plate"
    stack = src / "stack"
    masks = src / "masks" / "cell_mask_stack"
    stack.mkdir(parents=True)
    masks.mkdir(parents=True)
    _save_array_atomic(str(stack / "field1.npy"), np.zeros((4, 4, 1)))
    _save_array_atomic(str(stack / "field2.npy"), np.zeros((4, 4, 1)))
    _save_array_atomic(str(masks / "field1.npy"), np.zeros((4, 4)))
    (masks / "field2.npy").write_bytes(b"truncated")

    assert check_mask_folder(
        str(src), "cell_mask_stack", resume=False) is False
    assert check_mask_folder(
        str(src), "cell_mask_stack", resume=True) is True


# ---------------------------------------------------------------------------
# io.py wiring
# ---------------------------------------------------------------------------

class TestIoWiring:

    def test_merged_arrays_are_written_atomically(self, tmp_path):
        """os.replace means merged/<field>.npy is either absent or
        complete — never the prefix a killed np.save used to leave."""
        from spacr.io import _save_array_atomic
        target = tmp_path / 'merged' / 'plate1_A01_1.npy'
        array = np.arange(120, dtype=np.uint16).reshape(4, 5, 6)
        _save_array_atomic(str(target), array)

        assert np.array_equal(np.load(str(target)), array)
        assert validate_merged_field(str(target)) == (True, 'done')
        # No temporary files left behind.
        assert [p.name for p in target.parent.iterdir()] == \
            ['plate1_A01_1.npy']

    def test_atomic_write_leaves_the_old_file_when_it_fails(self, tmp_path):
        from spacr.io import _save_array_atomic
        target = tmp_path / 'plate1_A01_1.npy'
        good = np.ones((2, 2, 2), dtype=np.uint16)
        _save_array_atomic(str(target), good)

        class Unsavable:
            """np.save raises on this, mid-write."""
            def __array__(self, *args, **kwargs):
                raise RuntimeError('disk full')

        with pytest.raises(Exception):
            _save_array_atomic(str(target), Unsavable())
        assert np.array_equal(np.load(str(target)), good)
        assert list(target.parent.iterdir()) == [target]

    def test_load_and_concatenate_defaults_to_no_resume(self):
        """Opt-in: the default must be the pre-existing behaviour."""
        import inspect
        from spacr.io import _load_and_concatenate_arrays
        parameter = inspect.signature(
            _load_and_concatenate_arrays).parameters['resume']
        assert parameter.default is False

    @staticmethod
    def _build_src(root, fields=('plate1_A01_1', 'plate1_A01_2',
                                 'plate1_A01_3')):
        """A minimal src tree: stack/ plus one mask folder."""
        for sub in ('stack', 'masks/cell_mask_stack'):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        for stem in fields:
            np.save(os.path.join(root, 'stack', stem + '.npy'),
                    np.ones((6, 6, 2), dtype=np.uint16))
            np.save(os.path.join(root, 'masks', 'cell_mask_stack',
                                 stem + '.npy'),
                    np.ones((6, 6), dtype=np.uint16))
        return root

    def _merge(self, root, resume):
        from spacr.io import _load_and_concatenate_arrays
        _load_and_concatenate_arrays(root, [0, 1], 0, None, None, None,
                                     resume=resume)

    def test_resume_skips_complete_fields_and_redoes_the_rest(self, tmp_path):
        """The mask half of the deliverable, end to end."""
        root = self._build_src(str(tmp_path / 'exp'))
        merged = os.path.join(root, 'merged')

        self._merge(root, resume=False)
        assert sorted(os.listdir(merged)) == ['plate1_A01_1.npy',
                                              'plate1_A01_2.npy',
                                              'plate1_A01_3.npy']

        # Field 1: a valid stack with a recognisable value. If the resume
        # skips it, this survives; if it re-merges, it is overwritten.
        sentinel = np.full((6, 6, 3), 77, dtype=np.uint16)
        np.save(os.path.join(merged, 'plate1_A01_1.npy'), sentinel)
        # Field 2: what a crash mid-np.save leaves behind.
        truncate(os.path.join(merged, 'plate1_A01_2.npy'), 0.4)
        # Field 3: never got written at all.
        os.remove(os.path.join(merged, 'plate1_A01_3.npy'))

        self._merge(root, resume=True)

        assert np.array_equal(np.load(os.path.join(merged,
                                                   'plate1_A01_1.npy')),
                              sentinel)                      # skipped
        for stem in ('plate1_A01_2', 'plate1_A01_3'):        # redone
            path = os.path.join(merged, stem + '.npy')
            assert validate_merged_field(path) == (True, 'done')
            assert np.load(path).shape[-1] == 3

    def test_resume_off_redoes_every_field(self, tmp_path):
        """Default behaviour is untouched: without the flag the truncated
        file is simply overwritten, exactly as before."""
        root = self._build_src(str(tmp_path / 'exp'))
        merged = os.path.join(root, 'merged')
        self._merge(root, resume=False)

        sentinel = np.full((6, 6, 3), 77, dtype=np.uint16)
        np.save(os.path.join(merged, 'plate1_A01_1.npy'), sentinel)

        self._merge(root, resume=False)

        assert not np.array_equal(
            np.load(os.path.join(merged, 'plate1_A01_1.npy')), sentinel)

    def test_resume_reports_the_plan_before_merging(self, tmp_path, capsys):
        root = self._build_src(str(tmp_path / 'exp'))
        self._merge(root, resume=False)
        truncate(os.path.join(root, 'merged', 'plate1_A01_2.npy'), 0.4)
        capsys.readouterr()

        self._merge(root, resume=True)

        out = capsys.readouterr().out
        assert 'spaCR RESUME' in out
        assert REASON_TRUNCATED in out

    def test_core_passes_the_resume_setting_through(self):
        source = (Path(__file__).resolve().parent.parent / 'spacr' /
                  'core.py').read_text()
        assert "resume=settings.get('resume', False)" in source

    def test_save_settings_to_db_destroys_the_previous_record(self, tmp_path):
        """Pins the ordering constraint the measure_crop wiring depends on.

        ``_save_settings_to_db`` writes the ``settings`` table with
        ``if_exists='replace'``. Once it has run, the only record of what
        the *previous* run was configured to do is gone — so the resume's
        settings check has to happen before it, not after. If this test
        ever starts failing because the write became an append, the
        ordering requirement in measure_crop can be relaxed.
        """
        from spacr.io import _save_settings_to_db
        src = tmp_path / 'exp' / 'merged'
        os.makedirs(str(src))
        db = str(tmp_path / 'exp' / 'measurements' / 'measurements.db')

        _save_settings_to_db({'src': str(src), 'cell_mask_dim': 4})
        assert read_recorded_settings(db)['cell_mask_dim'] == '4'

        _save_settings_to_db({'src': str(src), 'cell_mask_dim': 5})
        recorded = read_recorded_settings(db)
        assert recorded['cell_mask_dim'] == '5'   # replaced, not appended
        assert len(recorded) == 2

    def test_measure_crop_orders_the_resume_before_the_settings_write(self):
        """Guards the applied measure.py wiring, once it is in place.

        The check is skipped while the wiring is still a proposed diff, so
        this test is safe to land ahead of it — but the moment
        ``plan_measure_resume`` appears in measure.py, it must appear
        *before* ``_save_settings_to_db``.
        """
        source = (Path(__file__).resolve().parent.parent / 'spacr' /
                  'measure.py').read_text()
        if 'plan_measure_resume(settings)' not in source:
            pytest.skip('measure.py resume wiring not applied yet')
        assert (source.index('plan_measure_resume(settings)')
                < source.index('_save_settings_to_db(settings)'))
        assert 'resume_plan.filter_files(files)' in source
