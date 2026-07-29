"""The ``png_list`` -> object-table join must not multiply a timelapse.

``io._read_and_join_tables`` joined ``png_list`` to ``cell`` on plate / row /
column / field only. On a non-timelapse database that key is unique and the
join is many-to-one. On a **timelapse** database every field carries T
timepoints, so the key matches T rows on each side: N objects x T frames came
back as N x T x T rows, with the crop from frame 1 attached to the object row
of frame 3. A 12-row database measured 36 rows out.

The sibling defect lives in ``utils._split_data``, which rebuilt ``prcf`` from
plate/row/column/field and dropped the timepoint that ``_map_wells`` had put
there — so ``prcfo``, the key every caller groups on, collapsed each object
across all of its frames and averaged them. That one loses rows rather than
inventing them: the same 12 rows came out as 4.

Every database here is built by the real writers — ``filepaths_to_database``
and ``_merge_and_save_to_database`` — from real spaCR file names. A
hand-rolled fixture that happens to match the schema is how this class of bug
survives an audit.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr.io import (JoinFanOut, MergeCardinalityError,
                      TimelapseKeyMismatch, _read_and_join_tables,
                      _read_and_merge_data)
from spacr.utils import (_merge_and_save_to_database, _split_data,
                         filepaths_to_database)

PLATE = 'plate1'
WELL = 'A01'


# ---------------------------------------------------------------------------
# database construction — real writers only
# ---------------------------------------------------------------------------

def _crop_name(field, time, obj):
    """The PNG file name ``measure_crop`` hands to ``filepaths_to_database``."""
    if time is None:
        return f'{PLATE}_{WELL}_{field}_{obj}.png'
    return f'{PLATE}_{WELL}_{field}_{time}_{obj}.png'


def _stem(field, time):
    """The stack file-name stem ``_merge_and_save_to_database`` parses."""
    if time is None:
        return f'{PLATE}_{WELL}_{field}'
    return f'{PLATE}_{WELL}_{field}_{time}'


def _build(root, fields=(1, 2), times=(1, 2, 3), objects=(1, 2),
           crops=True, cells=True, children=False,
           crop_timelapse=None, cell_timelapse=None):
    """Write a measurements.db with the real writers.

    ``times=(None,)`` means "not a timelapse": no timepoint appears in either
    the file names or the schema. ``crop_timelapse`` / ``cell_timelapse``
    override the flag handed to each writer independently, which is how the
    broken half-timelapse database of ``test_one_sided_time_column`` is made.
    ``children`` adds the cytoplasm and nucleus tables, which reach the result
    through the other two joins in the function.

    :returns: path to the database.
    """
    root = str(root)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)

    timelapse = times != (None,)
    if crop_timelapse is None:
        crop_timelapse = timelapse
    if cell_timelapse is None:
        cell_timelapse = timelapse

    if crops:
        crop_times = times if crop_timelapse else (None,)
        paths = [os.path.join(root, 'cell_png', _crop_name(f, t, o))
                 for f in fields for t in crop_times for o in objects]
        filepaths_to_database(paths, {'timelapse': crop_timelapse}, root, 'cell')

    if cells:
        cell_times = times if cell_timelapse else (None,)
        for f in fields:
            for t in cell_times:
                morph = pd.DataFrame({
                    'label': list(objects),
                    'cell_area': [100.0 * f + 10.0 * (t or 0) + o for o in objects],
                })
                intensity = pd.DataFrame({
                    'label': list(objects),
                    'cell_channel_0_mean_intensity': [float(o) for o in objects],
                })
                _merge_and_save_to_database(morph, intensity, 'cell', root,
                                            _stem(f, t), 'exp',
                                            timelapse=cell_timelapse)
                if not children:
                    continue
                cyto_morph = pd.DataFrame({
                    'label': list(objects),
                    'cytoplasm_area': [1000.0 * f + 10.0 * (t or 0) + o
                                       for o in objects],
                })
                _merge_and_save_to_database(
                    cyto_morph,
                    pd.DataFrame({'label': list(objects),
                                  'cytoplasm_channel_0_mean_intensity':
                                      [float(o) for o in objects]}),
                    'cytoplasm', root, _stem(f, t), 'exp',
                    timelapse=cell_timelapse)
                nuc_morph = pd.DataFrame({
                    'label': list(objects),
                    'cell_id': list(objects),        # one nucleus per cell
                    'nucleus_area': [10.0 * f + 1.0 * (t or 0) + o
                                     for o in objects],
                })
                _merge_and_save_to_database(
                    nuc_morph,
                    pd.DataFrame({'label': list(objects),
                                  'nucleus_channel_0_mean_intensity':
                                      [float(o) for o in objects]}),
                    'nucleus', root, _stem(f, t), 'exp',
                    timelapse=cell_timelapse)

    return os.path.join(root, 'measurements', 'measurements.db')


def _table(db, name):
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(f'SELECT * FROM {name}', conn)
    finally:
        conn.close()


def _columns(db, table):
    conn = sqlite3.connect(db)
    try:
        return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
    finally:
        conn.close()


def _legacy_join(db):
    """``_read_and_join_tables``'s png_list join *exactly* as it was before.

    Verbatim from the pre-fix source. It serves twice: the non-timelapse pin
    compares the new implementation against the old one rather than against a
    hand-written expectation of it, and the timelapse test runs it on the same
    database to show the fan-out it used to produce.
    """
    conn = sqlite3.connect(db)
    try:
        cell = pd.read_sql('SELECT * FROM cell', conn)
        png = pd.read_sql('SELECT * FROM png_list', conn)
    finally:
        conn.close()
    png_list_df = png[['cell_id', 'png_path', 'plateID', 'rowID', 'columnID',
                       'fieldID']].copy()
    png_list_df['cell_id'] = png_list_df['cell_id'].str[1:].astype(int)
    png_list_df.rename(columns={'cell_id': 'object_label'}, inplace=True)
    join_cols = ['object_label', 'plateID', 'rowID', 'columnID', 'fieldID']
    return pd.merge(cell, png_list_df, on=join_cols, how='left')


# ---------------------------------------------------------------------------
# the defect
# ---------------------------------------------------------------------------

def test_timelapse_join_returns_one_row_per_object_per_frame(tmp_path):
    """2 fields x 3 frames x 2 cells -> 12 rows, not 36, with the right crop.

    The row count alone is not enough: a wrong-key merge can also return the
    right number of rows with nothing merged in, so every merged column is
    checked for content and every crop is checked against the timepoint of the
    row it landed on.
    """
    db = _build(tmp_path)
    cell = _table(db, 'cell')
    png = _table(db, 'png_list')
    assert len(cell) == 12 and len(png) == 12
    assert 'timeID' in cell.columns and 'timeID' in png.columns

    # The pre-fix join, run side by side on the same database, to keep the
    # size of the defect in the record rather than in a commit message.
    assert len(_legacy_join(db)) == 36         # 12 x T

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert len(out) == 12
    assert len(out) == len(cell)
    # Nothing was left unmatched — the merged column is populated on every row.
    assert out['png_path'].notna().all()
    assert out['png_path'].nunique() == 12
    # And each crop sits on the row for its OWN timepoint and object.
    for _, row in out.iterrows():
        name = row['png_path'].rsplit('/', 1)[-1]
        field, time, obj = name[:-len('.png')].split('_')[2:]
        assert f'f{field}' == row['fieldID']
        assert f't{time}' == row['timeID']
        assert int(obj) == row['object_label']
    # The measurement stayed attached to its own frame.
    assert set(out['cell_area']) == {100.0 * f + 10.0 * t + o
                                     for f in (1, 2) for t in (1, 2, 3)
                                     for o in (1, 2)}


def test_timelapse_join_of_every_object_table_stays_one_row_per_frame(tmp_path):
    """cell + cytoplasm + nucleus + png_list, all four, still 12 rows.

    Only the ``png_list`` join carries the timepoint explicitly; cytoplasm and
    nucleus come in on ``prcf``, which ``_map_wells(timelapse=True)`` already
    writes as plate_row_column_field_TIME. This is what says so out loud, and
    it would catch a future "simplification" of prcf back to four components.
    """
    db = _build(tmp_path, children=True)

    out = _read_and_join_tables(db)

    assert len(out) == 12
    assert out['png_path'].notna().all()
    assert out['cytoplasm_area'].notna().all()
    assert out['nucleus_area'].notna().all()
    assert out['count_nucleus'].eq(1).all()
    # Every table's value is the one measured on that row's own frame.
    for _, row in out.iterrows():
        field = int(row['fieldID'][1:])
        time = int(row['timeID'][1:])
        obj = row['object_label']
        assert row['cell_area'] == 100.0 * field + 10.0 * time + obj
        assert row['cytoplasm_area'] == 1000.0 * field + 10.0 * time + obj
        assert row['nucleus_area'] == 10.0 * field + 1.0 * time + obj


def test_single_timepoint_database_still_joins_on_time(tmp_path):
    """T=1 looks like a non-timelapse run but is not one.

    With one frame the time-blind join happens to give the right answer, so
    this case cannot detect the bug — what it pins is that the timepoint is
    still part of the key (the column survives the join and carries its value)
    rather than being dropped because it "made no difference".
    """
    db = _build(tmp_path, times=(1,))

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert len(out) == 4                       # 2 fields x 1 frame x 2 cells
    assert out['png_path'].notna().all()
    assert set(out['timeID']) == {'t1'}
    # One timeID column, not a timeID_x / timeID_y pair: it was a join key.
    assert [c for c in out.columns if c.startswith('time')] == ['timeID']


def test_non_timelapse_join_is_byte_identical_to_the_old_implementation(tmp_path):
    """The pin: with no timepoint anywhere, nothing about the join changed."""
    db = _build(tmp_path, times=(None,))
    assert 'timeID' not in _columns(db, 'cell')
    assert 'timeID' not in _columns(db, 'png_list')

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])
    expected = _legacy_join(db)

    pd.testing.assert_frame_equal(out, expected)
    assert len(out) == 4                       # 2 fields x 2 cells
    assert out['png_path'].notna().all()


def test_non_timelapse_join_with_fewer_crops_than_cells_is_not_a_fan_out(tmp_path):
    """Crops are allowed to be a strict subset of the measured objects.

    ``len(result) > len(png_list)`` is therefore NOT the fan-out condition —
    this database trips it while being perfectly healthy. The condition that
    holds is ``len(result) == len(cell)``.
    """
    db = _build(tmp_path, times=(None,), crops=False)
    # One crop for one of the four cells, as if save_png had been off for the
    # rest of the run.
    filepaths_to_database([str(tmp_path / 'cell_png' / _crop_name(1, None, 1))],
                          {'timelapse': False}, str(tmp_path), 'cell')

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert len(_table(db, 'png_list')) == 1
    assert len(out) == 4                       # every cell survives
    assert out['png_path'].notna().sum() == 1  # only one of them has a crop


# ---------------------------------------------------------------------------
# legacy spellings
# ---------------------------------------------------------------------------

def _downgrade_png_list_to_time_id(db):
    """Undo the timeID rename, leaving the database a pre-fix spaCR wrote."""
    conn = sqlite3.connect(db)
    try:
        conn.execute('ALTER TABLE png_list RENAME COLUMN "timeID" TO "time_id"')
        conn.commit()
    finally:
        conn.close()
    assert 'time_id' in _columns(db, 'png_list')


def test_legacy_png_list_spelling_reads_without_manual_migration(tmp_path):
    """A database still saying ``time_id`` joins correctly on first read."""
    db = _build(tmp_path)
    _downgrade_png_list_to_time_id(db)

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert len(out) == 12
    assert out['png_path'].notna().all()
    # Repaired in place on the way through, so the second read is a normal one.
    assert 'timeID' in _columns(db, 'png_list')
    assert 'time_id' not in _columns(db, 'png_list')


def test_reader_tolerates_the_two_spellings_when_the_migration_did_not_run(
        tmp_path, monkeypatch):
    """Readers stay tolerant even when nothing repaired the schema first.

    ``rename_columns_in_db`` normally unifies the spellings before the join
    ever sees them, but it needs a writable database — on a read-only copy, or
    on a table that carries both spellings at once, the two sides can still
    disagree at read time. Neutering the migration is the cheapest way to hold
    the reader to that promise on its own.
    """
    db = _build(tmp_path)
    _downgrade_png_list_to_time_id(db)
    import spacr.utils
    monkeypatch.setattr(spacr.utils, 'rename_columns_in_db',
                        lambda path: [])

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert 'time_id' in _columns(db, 'png_list')   # genuinely not migrated
    assert len(out) == 12
    assert out['png_path'].notna().all()
    # The object table's own spelling is what survives into the result; the
    # png_list copy was aligned to it, not the other way round.
    assert 'timeID' in out.columns
    assert 'time_id' not in out.columns


# ---------------------------------------------------------------------------
# the loud failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('crop_tl,cell_tl,png_col,cell_col',
                         [(True, False, 'timeID', None),
                          (False, True, None, 'timeID')])
def test_one_sided_time_column_reports_instead_of_fanning_out(
        tmp_path, crop_tl, cell_tl, png_col, cell_col):
    """Half a timelapse is a broken database, not a join to attempt."""
    db = _build(tmp_path, crop_timelapse=crop_tl, cell_timelapse=cell_tl)
    assert ('timeID' in _columns(db, 'png_list')) is crop_tl
    assert ('timeID' in _columns(db, 'cell')) is cell_tl

    with pytest.raises(TimelapseKeyMismatch) as excinfo:
        _read_and_join_tables(db, table_names=['cell', 'png_list'])

    message = str(excinfo.value)
    assert 'png_list' in message and 'cell' in message
    assert repr(png_col) in message and repr(cell_col) in message


def test_duplicate_png_list_rows_are_reported_as_a_fan_out(tmp_path):
    """The crop step run twice doubles png_list; the join must say so.

    This is the only way left for the join to grow the object table, and it is
    a real one — ``filepaths_to_database`` appends, so a re-run adds a second
    row for every crop.
    """
    db = _build(tmp_path, times=(None,))
    paths = [os.path.join(str(tmp_path), 'cell_png', _crop_name(f, None, o))
             for f in (1, 2) for o in (1, 2)]
    filepaths_to_database(paths, {'timelapse': False}, str(tmp_path), 'cell')
    assert len(_table(db, 'png_list')) == 8

    with pytest.raises(JoinFanOut) as excinfo:
        _read_and_join_tables(db, table_names=['cell', 'png_list'])

    message = str(excinfo.value)
    assert 'png_list' in message and 'cell' in message
    assert "validate='one_to_one'" in message
    assert 'png_list has duplicated' in message
    assert "['object_label', 'plateID', 'rowID', 'columnID', 'fieldID']" in message


def test_duplicate_cell_keys_are_reported_before_crop_join(tmp_path):
    """A repeated measurement row violates the left side of the 1:1 join."""
    db = _build(tmp_path, times=(None,))
    with sqlite3.connect(db) as conn:
        columns = [row[1] for row in conn.execute(
            'PRAGMA table_info("cell")')]
        quoted = ', '.join(f'"{column}"' for column in columns)
        conn.execute(
            f'INSERT INTO "cell" ({quoted}) SELECT {quoted} FROM "cell" '
            'WHERE object_label = 1 LIMIT 1'
        )

    with pytest.raises(MergeCardinalityError) as excinfo:
        _read_and_join_tables(db, table_names=['cell', 'png_list'])

    message = str(excinfo.value)
    assert "validate='one_to_one'" in message
    assert 'cell has duplicated' in message
    assert 'png_list has duplicated' not in message


def test_png_list_without_a_cell_table_keeps_its_timepoint(tmp_path):
    """The no-cell-table escape hatch returns the timepoint too.

    Without it the caller gets one row per object per frame and no way to tell
    the frames apart.
    """
    db = _build(tmp_path, cells=False)

    out = _read_and_join_tables(db, table_names=['cell', 'png_list'])

    assert list(out.columns) == ['object_label', 'png_path', 'plateID',
                                 'rowID', 'columnID', 'fieldID', 'timeID']
    assert len(out) == 12
    assert set(out['timeID']) == {'t1', 't2', 't3'}


# ---------------------------------------------------------------------------
# R1: _split_data must keep the timepoint in prcf / prcfo
# ---------------------------------------------------------------------------

def test_split_data_keeps_the_timepoint_in_prcf(tmp_path):
    """12 object-rows across 3 frames stay 12 groups, not 4.

    ``_map_wells(timelapse=True)`` writes ``plate_row_column_field_time`` into
    the database's own ``prcf``; the rebuild must agree with it.
    """
    db = _build(tmp_path)
    cells = _table(db, 'cell')
    cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
    cells = cells.assign(prcfo=lambda x: x['prcf'] + '_' + x['object_label'])
    assert cells['prcfo'].nunique() == 12
    # The key the rebuild used to produce — plate/row/column/field/object with
    # the timepoint dropped — has only 4 distinct values for these 12 rows.
    time_blind = (cells['plateID'] + '_' + cells['rowID'] + '_' +
                  cells['columnID'] + '_' + cells['fieldID'] + '_' +
                  cells['object_label'])
    assert time_blind.nunique() == 4

    numeric, metadata = _split_data(cells, 'prcfo', 'object_label')

    assert len(numeric) == 12                  # was 4: frames averaged together
    assert set(numeric.index) == set(cells['prcfo'])
    assert set(metadata['prcf']) == {f'{PLATE}_r1_c1_f{f}_t{t}'
                                     for f in (1, 2) for t in (1, 2, 3)}
    # prcft was always the timepoint key; prcf now agrees with it.
    assert (metadata['prcf'] == metadata['prcft']).all()
    # Each group is a single row, so the "mean" is the original value.
    assert set(numeric['cell_area']) == {100.0 * f + 10.0 * t + o
                                         for f in (1, 2) for t in (1, 2, 3)
                                         for o in (1, 2)}


def test_split_data_prcf_is_unchanged_without_a_timepoint(tmp_path):
    """No time column -> the same plate_row_column_field key as always."""
    db = _build(tmp_path, times=(None,))
    cells = _table(db, 'cell')
    cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))

    numeric, metadata = _split_data(cells, 'prcfo', 'object_label')

    assert len(numeric) == 4
    assert set(metadata['prcf']) == {f'{PLATE}_r1_c1_f1', f'{PLATE}_r1_c1_f2'}
    assert 'prcft' not in metadata.columns
    assert set(numeric.index) == {f'{PLATE}_r1_c1_f{f}_o{o}'
                                  for f in (1, 2) for o in (1, 2)}


def test_read_and_merge_data_keeps_the_frames_apart(tmp_path):
    """End to end: the two fixes together give one row per object per frame."""
    db = _build(tmp_path)

    merged, obj_dfs = _read_and_merge_data([db], ['cell', 'png_list'],
                                           nuclei_limit=None,
                                           pathogen_limit=None)

    assert len(merged) == 12
    assert set(merged.index) == {f'{PLATE}_r1_c1_f{f}_t{t}_o{o}'
                                 for f in (1, 2) for t in (1, 2, 3)
                                 for o in (1, 2)}
    # png_list came along on the same key, one distinct crop per row.
    assert merged['png_path'].notna().all()
    assert merged['png_path'].nunique() == 12
    # The index agrees with the prcfo the crop writer put in the database.
    # plot.py's join_measurments_and_annotation merges this frame with
    # png_list on exactly that column, and the two used to disagree by the
    # timepoint, so on a timelapse it matched nothing at all.
    assert set(merged.index) == set(_table(db, 'png_list')['prcfo'])
    # Nothing was dropped on the way: every object row of the cell table is
    # represented exactly once.
    assert len(obj_dfs[0]) == 12
    assert merged['prcf'].nunique() == 6        # 2 fields x 3 frames
