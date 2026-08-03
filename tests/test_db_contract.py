"""Every writer and every reader of the project database must agree.

The database is the one thing every module touches. ``measure`` writes the
object tables, ``utils.filepaths_to_database`` writes ``png_list``, ``io``
writes ``settings``/``object_counts``, ``errors`` writes ``run_status``,
``foreign`` imports into the same tables, ``resume`` deletes from them, and
``predictions``/``agreement``/``active_learning``/``ml`` read them. None of
those modules imports the others, so the only thing keeping them on one
schema is that :mod:`spacr.schema` declares it and everybody obeys.

Nothing checked that. A database contract audit found the drift by hand --
``column_name`` against ``columnID``, ``time_id`` against ``timeID``,
``cell_id`` meaning TEXT in one table and REAL in another -- and the fixes
landed, but the audit was a document. A document does not run, so the same
class of bug can come back silently. This is the audit as tests.

Two bugs this area has actually produced, both of which have a *general*
form that this file pins and neither of which any hand-built database would
have shown:

1. **``rowid`` shadowing.** Every spaCR object table declares a column
   called ``rowID`` -- the plate row, ``'r1'``. SQLite identifiers are
   case-insensitive and a declared column always shadows the implicit
   ``rowid``, so ``SELECT rowid FROM cell`` returns ``'r1'`` once per row
   and ``DELETE ... WHERE rowid IN (...)`` deletes by *plate row*. Measured
   below on a table the real writer filled: a DELETE that targeted two rows
   removed all six.

2. **Key-column collision.** The obvious repair -- delete by the declared
   key, :meth:`spacr.schema.ObjectTableSchema.row_key_columns` -- is
   *equally* destructive, because the writer appends: two rows for the same
   object can share all five key columns. Measured below: a keyed delete
   aimed at one object removed two rows. So "a keyed delete is safe" is a
   false property and no test may assert it. The property that is true, and
   the one asserted here, is that a delete removes exactly the number the
   predicate that cleared it counted, or rolls back.

Every database in this file is built by spaCR's own writers --
:func:`spacr.utils._merge_and_save_to_database`,
:func:`spacr.utils.filepaths_to_database`,
:func:`spacr.io._save_object_counts_to_database`,
:func:`spacr.io._save_settings_to_db` and
:meth:`spacr.errors.RunLedger.stamp` -- because building ``png_list`` by
hand, without the ``rowID`` column, is precisely how bug 1 stayed hidden.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import schema
from spacr.database_schema import (CURRENT_SCHEMA_VERSION,
                                   DatabaseSchemaTooNewError)


# ---------------------------------------------------------------------------
# building a project the way spaCR builds one
# ---------------------------------------------------------------------------

#: Object tables the writers fill here. ``organelle`` is owned but optional
#: and has no canonical contract, so it is exercised separately where it
#: matters rather than assumed present.
BUILT_OBJECT_TABLES = ('cell', 'cytoplasm', 'nucleus', 'pathogen')

N_OBJECTS = 3


def measure_field(root, stem, tables=BUILT_OBJECT_TABLES, n_objects=N_OBJECTS,
                  timelapse=False):
    """Write one field's object rows through the real measure writer.

    This is the same call :func:`spacr.measure.measure_crop` makes, with the
    same shape of frames: morphology and intensity keyed on ``label``, and a
    float ``cell_id`` on the child tables because ``measure`` writes NaN
    there for "no overlapping cell".
    """
    labels = list(range(1, n_objects + 1))
    for table in tables:
        morphology = pd.DataFrame({
            'label': labels,
            f'{table}_area': [100.0 + i for i in range(n_objects)]})
        intensity = pd.DataFrame({
            'label': labels,
            f'{table}_channel_0_mean_intensity': [5.0] * n_objects})
        if table in schema.CHILD_OBJECT_TABLES:
            morphology['cell_id'] = np.asarray(labels, dtype=float)
        from spacr.utils import _merge_and_save_to_database
        _merge_and_save_to_database(morphology, intensity, table, root, stem,
                                    'exp', timelapse)


def write_crops(root, stem, n=N_OBJECTS, crop_mode='cell', timelapse=False):
    """Index crop file names through the real ``filepaths_to_database``."""
    from spacr.utils import filepaths_to_database
    folder = os.path.join(root, 'data', f'{crop_mode}_png')
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f'{stem}_{i + 1}.png') for i in range(n)]
    filepaths_to_database(paths, {'timelapse': timelapse}, root, crop_mode)
    return paths


def db_of(root):
    return os.path.join(root, 'measurements', 'measurements.db')


def build_project(root, fields=('plate1_A01_1', 'plate1_A01_2'),
                  bookkeeping=True):
    """A project database with every writer that touches ``measurements.db``."""
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    os.makedirs(os.path.join(root, 'data'), exist_ok=True)
    for stem in fields:
        measure_field(root, stem)
        write_crops(root, stem)
    db = db_of(root)
    if bookkeeping:
        from spacr.errors import RunLedger
        from spacr.io import (_save_object_counts_to_database,
                              _save_settings_to_db)
        _save_object_counts_to_database(
            [np.array([[0, 1], [2, 3]])], 'cell',
            [f'{fields[0]}.npy'], db, '')
        # _save_settings_to_db writes to dirname(src)/measurements, so src is
        # the data folder of the project rather than the project itself.
        _save_settings_to_db({'src': os.path.join(root, 'data'),
                              'stage': 'measure', 'experiment': 'exp'})
        RunLedger(name='measure').stamp(db)
    return db


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    """One real project database, shared by every read-only assertion here."""
    root = str(tmp_path_factory.mktemp('contract_project'))
    return build_project(root)


@pytest.fixture()
def fresh_project(tmp_path):
    """A private project for tests that write to or delete from it."""
    root = str(tmp_path / 'project')
    return build_project(root)


# ---------------------------------------------------------------------------
# small SQLite helpers -- everything here asks the database, never the source
# ---------------------------------------------------------------------------

def table_names(db):
    con = sqlite3.connect(db)
    try:
        return sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"))
    finally:
        con.close()


def declared_columns(db, table):
    """``{column: declared SQLite type}`` straight from ``PRAGMA``."""
    con = sqlite3.connect(db)
    try:
        return {r[1]: r[2] for r in con.execute(f'PRAGMA table_info("{table}")')}
    finally:
        con.close()


def stored_types(db, table, column):
    """The distinct run-time storage classes actually present in a column.

    A declared type is an affinity, not a guarantee; this is what the rows
    really hold, which is what a reader's ``.astype`` or comparison meets.
    """
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute(
            f'SELECT DISTINCT typeof("{column}") FROM "{table}"')}
    finally:
        con.close()


def row_count(db, table):
    con = sqlite3.connect(db)
    try:
        return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        con.close()


# ===========================================================================
# 1. Every table a writer creates matches what spacr/schema.py declares
# ===========================================================================

def test_the_writers_create_the_tables_the_schema_declares(project):
    """What the writers made is what :data:`schema.OWNED_TABLES` names.

    Stated as a subset in this direction -- an optional table nobody
    switched on is legitimately absent. The other direction, "nothing was
    created that the schema does not declare", is the interesting one and
    is its own test below.
    """
    created = set(table_names(project))
    assert created, 'the writers produced no tables at all'
    for table in BUILT_OBJECT_TABLES + ('png_list', 'settings',
                                        'object_counts'):
        assert table in created, table
        assert table in schema.OWNED_TABLES, (
            f'{table!r} was created by a spaCR writer but is not declared in '
            f'schema.OWNED_TABLES')


@pytest.mark.xfail(
    strict=True,
    reason=(
        'spacr/schema.py:1316 BOOKKEEPING_TABLES omits two tables spaCR '
        'writes into measurements.db itself: settings_history '
        '(spacr/io.py:2769, CREATE TABLE IF NOT EXISTS, written by '
        '_save_settings_to_db) and run_status (spacr/errors.py:595, written '
        'by RunLedger._stamp_db). OWNED_TABLES is documented at '
        'spacr/schema.py:1318 as "Every table spaCR creates in a '
        'measurements database. A table not in here was put there by someone '
        'else and must be left alone by any migration" -- so the declaration '
        'is false for these two. Consequences: (a) schema.table_key_columns '
        'raises KeyParseError for both, so no caller can ask for their key '
        'columns; (b) spacr/doctor.py:1588 computes '
        '"names & set(OWNED_TABLES)", so it undercounts every project '
        'database and, on one holding only run_status, reports "contains '
        "none of spaCR's tables ... probably not a measurements database\" "
        '-- measured, see the companion xfail below. schema.py is owned by '
        'another agent in this session, so the fix (adding both names to '
        'BOOKKEEPING_TABLES, and giving table_key_columns a key for them: '
        "settings_history keys on ('run_id','stage','setting_key'), "
        "run_status on ('run_id',)) is reported rather than applied."),
)
def test_every_table_a_spacr_writer_creates_is_declared_owned(project):
    """Nothing spaCR writes into its own database is a stranger to the schema.

    This is the direction that matters for migrations and for the doctor: a
    table absent from ``OWNED_TABLES`` is, by that constant's own
    documentation, somebody else's -- so it is skipped, left alone, and
    reported as foreign. Two of spaCR's own bookkeeping tables are in that
    position.
    """
    created = set(table_names(project))
    undeclared = sorted(created - set(schema.OWNED_TABLES))
    assert undeclared == [], (
        f'spaCR writers created {undeclared}, which schema.OWNED_TABLES does '
        f'not declare; OWNED_TABLES says anything not in it belongs to '
        f'someone else')


@pytest.mark.xfail(
    strict=True,
    reason=(
        'The user-visible half of the OWNED_TABLES gap above. A run that '
        'dies before it measures anything leaves a measurements.db holding '
        'only run_status (spacr/errors.py:595). spacr/doctor.py:1588 '
        'intersects the table names with schema.OWNED_TABLES, which does '
        'not contain run_status, so the intersection is empty and '
        'doctor.py:1589-1602 reports WARN "is a valid SQLite file but '
        "contains none of spaCR's tables (1 tables found). This is probably "
        'not a measurements database." and advises pointing --db elsewhere. '
        'It is spaCR\'s own database, spaCR wrote the table, and '
        'errors.read_run_status reads that same file happily and names the '
        'stage that failed -- so the doctor sends a user away from the one '
        'file that explains their failed run. Fixed by declaring run_status '
        'and settings_history in schema.BOOKKEEPING_TABLES; schema.py is '
        'owned by another agent in this session.'),
)
def test_the_doctor_recognises_a_database_holding_only_a_run_stamp(tmp_path):
    """spaCR must recognise a database it wrote itself."""
    from pathlib import Path

    from spacr.doctor import _database_table_rows
    from spacr.errors import RunLedger, read_run_status
    from spacr.io import _create_database

    meas = tmp_path / 'exp' / 'measurements'
    meas.mkdir(parents=True)
    db = str(meas / 'measurements.db')
    _create_database(db)
    RunLedger(name='preprocess').stamp(db)

    assert table_names(db) == ['run_status']
    # The file is readable by spaCR's own reader, so it is plainly spaCR's.
    assert [row['name'] for row in read_run_status(db)] == ['preprocess']

    results = _database_table_rows(Path(db))
    assert results
    assert all(r.status != 'WARN' for r in results), (
        f'the doctor disowned a database spaCR wrote: '
        f'{[r.message for r in results]}')


@pytest.mark.parametrize('table', schema.CANONICAL_OBJECT_TABLES)
def test_a_written_object_table_carries_every_required_column(project, table):
    """The writer's output satisfies ``OBJECT_TABLE_REQUIRED_COLUMNS``."""
    columns = declared_columns(project, table)
    missing = [c for c in schema.OBJECT_TABLE_REQUIRED_COLUMNS
               if c not in columns]
    assert missing == [], f'{table} is missing {missing}'


@pytest.mark.parametrize('table', schema.CANONICAL_OBJECT_TABLES)
def test_the_parent_link_is_present_exactly_where_the_schema_says(project,
                                                                  table):
    """``cell_id`` on the child tables, and on no other.

    A parent table that grew a ``cell_id`` would join to itself; a child
    table that lost one silently drops every roll-up onto the cell.
    """
    contract = schema.object_table_schema(table)
    columns = declared_columns(project, table)
    if contract.parent_column is None:
        assert 'cell_id' not in columns, (
            f'{table} is a parent table but declares cell_id')
    else:
        assert contract.parent_column in columns, (
            f'{table} is a child table and must declare '
            f'{contract.parent_column}')


@pytest.mark.parametrize('table', schema.CANONICAL_OBJECT_TABLES)
def test_the_declared_key_columns_exist_in_the_written_table(project, table):
    """``row_key_columns`` and ``table_key_columns`` name real columns.

    Both are consumed as SQL identifiers -- ``resume.clear_field_rows``
    interpolates ``table_key_columns`` straight into a DELETE -- so a name
    the table does not have is a runtime SQL error at the worst moment.
    """
    columns = set(declared_columns(project, table))
    contract = schema.object_table_schema(table)
    for key in contract.row_key_columns():
        assert key in columns, f'{table}.{key} declared by schema, absent on disk'
    for key in schema.table_key_columns(table):
        assert key in columns, f'{table}.{key} declared by schema, absent on disk'


def test_the_crop_tables_declared_key_exists_too(project):
    """``png_list`` keys on ``prcfo``, not on an object label."""
    columns = set(declared_columns(project, 'png_list'))
    for key in schema.table_key_columns('png_list'):
        assert key in columns, f'png_list.{key} declared by schema, absent on disk'
    assert schema.OBJECT_LABEL_KEY not in columns, (
        'png_list must not carry object_label: its object id is the TEXT '
        "'o<N>' in cell_id, and a reader that found object_label here would "
        'join the two on incompatible types')


@pytest.mark.parametrize('table', BUILT_OBJECT_TABLES + ('png_list',))
def test_every_measurement_table_carries_the_field_key(project, table):
    """The four field key columns are what every module joins on.

    ``resume.discover_field_tables`` uses their presence to decide a table
    is per-field; ``foreign`` writes them so its import joins to the
    measurements; ``io`` merges on them. One table spelling them differently
    is a silent empty join.
    """
    columns = set(declared_columns(project, table))
    missing = [c for c in schema.FIELD_KEY_COLUMNS if c not in columns]
    assert missing == [], f'{table} is missing field key columns {missing}'


# ===========================================================================
# 2. The types readers assume are the types the writers declare and store
# ===========================================================================

#: What the on-disk contract is, per table and column: the declared affinity
#: and the storage classes a reader may actually meet. Asserted against a
#: database the real writers filled, so this table is checked rather than
#: believed.
TYPE_CONTRACT = (
    ('cell', 'object_label', 'INTEGER', {'integer'}),
    ('cell', 'plateID', 'TEXT', {'text'}),
    ('cell', 'rowID', 'TEXT', {'text'}),
    ('cell', 'columnID', 'TEXT', {'text'}),
    ('cell', 'fieldID', 'TEXT', {'text'}),
    ('cell', 'prcf', 'TEXT', {'text'}),
    ('cell', 'file_name', 'TEXT', {'text'}),
    ('cell', 'path_name', 'TEXT', {'text'}),
    ('nucleus', 'object_label', 'INTEGER', {'integer'}),
    # REAL, not INTEGER: measure writes NaN for "no overlapping cell", and a
    # column that must hold NaN cannot have integer affinity.
    ('nucleus', 'cell_id', 'REAL', {'real'}),
    ('pathogen', 'cell_id', 'REAL', {'real'}),
    ('png_list', 'png_path', 'TEXT', {'text'}),
    ('png_list', 'file_name', 'TEXT', {'text'}),
    ('png_list', 'prcfo', 'TEXT', {'text'}),
    # The one name that means two types in one database. png_list.cell_id is
    # the TEXT 'o<N>' tail of prcfo; nucleus.cell_id is the REAL parent
    # label. Pinned in both directions so neither can quietly become the
    # other -- a SQL join between them matches nothing at all.
    ('png_list', 'cell_id', 'TEXT', {'text'}),
)


@pytest.mark.parametrize('table, column, affinity, storage', TYPE_CONTRACT)
def test_the_column_type_a_reader_assumes_is_the_type_on_disk(
        project, table, column, affinity, storage):
    declared = declared_columns(project, table)
    assert column in declared, f'{table}.{column} does not exist'
    assert declared[column] == affinity, (
        f'{table}.{column} is declared {declared[column]}, readers assume '
        f'{affinity}')
    present = stored_types(project, table, column) - {'null'}
    assert present <= storage, (
        f'{table}.{column} stores {sorted(present)}, readers assume '
        f'{sorted(storage)}')


#: Every column a real reader names in a real ``SELECT``, with the module
#: and function that names it. Taken from the SELECT statements themselves,
#: and asserted against a database the writers filled -- so a writer that
#: drops or respells a column fails here rather than in a user's run, where
#: the symptom is an empty merge rather than an error.
READER_SELECTS = (
    ('measure.generate_object_dataset', 'cell',
     ('object_label', 'path_name', 'plateID', 'rowID', 'columnID', 'fieldID')),
    ('measure.get_object_counts', 'object_counts',
     ('count_type', 'object_count')),
    ('io.crop_rows_from_object_table', 'cell',
     ('object_label', 'plateID', 'rowID', 'columnID', 'fieldID', 'prcf',
      'file_name', 'path_name')),
    ('io._merged_field_paths', 'cell',
     ('plateID', 'rowID', 'columnID', 'fieldID', 'path_name', 'file_name')),
    ('io.crop_rows_from_object_table (parent map)', 'nucleus',
     ('object_label', 'prcf', 'cell_id')),
    ('io._settings_history_rows', 'settings_history',
     ('run_id', 'stage', 'stamped_utc', 'setting_key', 'setting_value')),
    ('io._save_settings_to_db / resume.read_recorded_settings', 'settings',
     ('setting_key', 'setting_value')),
    ('crops.crop_settings_from_db', 'settings',
     ('setting_key', 'setting_value')),
    ('resume.completed_fields_in_db', 'cell',
     ('plateID', 'rowID', 'columnID', 'fieldID')),
    ('agreement.load_annotations', 'png_list', ('png_path',)),
    ('active_learning.build_queue', 'png_list', ('png_path',)),
    ('predictions._merge_locked', 'png_list',
     ('prcfo', 'png_path', 'file_name')),
    ('foreign._holds_any_field / _write_view', 'cell',
     ('prcf', 'object_label')),
    ('errors.read_run_status', 'run_status',
     ('run_id', 'name', 'status', 'n_attempted', 'n_succeeded', 'n_failed')),
)


@pytest.mark.parametrize('reader, table, columns', READER_SELECTS,
                         ids=[f'{r}-{t}' for r, t, _ in READER_SELECTS])
def test_every_column_a_reader_selects_exists_on_disk(project, reader, table,
                                                      columns):
    """A reader's SELECT list against the columns the writers really wrote."""
    assert table in table_names(project), (
        f'{reader} reads {table!r}, which no writer created')
    declared = set(declared_columns(project, table))
    missing = [c for c in columns if c not in declared]
    assert missing == [], (
        f'{reader} selects {missing} from {table}, which does not have them; '
        f'it has {sorted(declared)}')


@pytest.mark.parametrize('reader, table, columns', READER_SELECTS,
                         ids=[f'{r}-{t}' for r, t, _ in READER_SELECTS])
def test_the_select_a_reader_issues_actually_runs(project, reader, table,
                                                  columns):
    """The SELECT is executed, not just spelled -- and returns the rows.

    Existence of a column is necessary and not sufficient: a reader that
    names a column of a table that is always empty gets an answer that looks
    like "nothing matched".
    """
    quoted = ', '.join(f'"{c}"' for c in columns)
    con = sqlite3.connect(project)
    try:
        rows = con.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
    finally:
        con.close()
    assert rows, f'{reader}: SELECT ... FROM {table} returned no rows'
    assert all(len(r) == len(columns) for r in rows)


def test_object_label_is_an_int_everywhere_a_reader_casts_it(project):
    """``int(row['object_label'])`` must not meet a string or a float.

    ``measure.generate_object_dataset`` and ``io.crop_rows_from_object_table``
    both do exactly that cast and then use the result to name a crop file.
    A float would name it ``obj1.0``.
    """
    con = sqlite3.connect(project)
    try:
        for table in BUILT_OBJECT_TABLES:
            kinds = {r[0] for r in con.execute(
                f'SELECT DISTINCT typeof("object_label") FROM "{table}"')}
            assert kinds == {'integer'}, f'{table}.object_label holds {kinds}'
    finally:
        con.close()


def test_prcf_is_exactly_the_field_key_joined_so_both_join_styles_agree(
        project):
    """The two ways modules identify a field must denote the same field.

    ``io._read_and_join_tables`` joins the cytoplasm on ``prcf``;
    ``resume`` and the ``png_list`` join key on the four field columns; and
    ``foreign``'s ``<object>_with_foreign`` view joins on
    ``(prcf, object_label)``. If ``prcf`` ever stops being those four columns
    joined by ``_``, the view and the merge silently disagree about which
    rows are the same object. Checked in SQL, on the writers' output.
    """
    sep = schema.KEY_SEPARATOR
    expression = f" || '{sep}' || ".join(
        f'"{c}"' for c in schema.FIELD_KEY_COLUMNS)
    con = sqlite3.connect(project)
    try:
        for table in BUILT_OBJECT_TABLES:
            bad = con.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f'WHERE "prcf" IS NOT {expression}').fetchone()[0]
            assert bad == 0, (
                f'{table}: {bad} row(s) whose prcf is not '
                f'{"_".join(schema.FIELD_KEY_COLUMNS)}')
    finally:
        con.close()


def test_prcfo_is_the_field_key_plus_the_object_so_crops_join_to_objects(
        project):
    """``png_list.prcfo`` must be its own field key plus its object id.

    ``predictions`` keys its merge on ``prcfo`` and falls back to rebuilding
    it from the metadata columns; the two must produce the same string or a
    re-run of a stage silently scores a different set of crops.
    """
    sep = schema.KEY_SEPARATOR
    parts = list(schema.FIELD_KEY_COLUMNS) + ['cell_id']
    expression = f" || '{sep}' || ".join(f'"{c}"' for c in parts)
    con = sqlite3.connect(project)
    try:
        bad = con.execute(
            f'SELECT COUNT(*) FROM "png_list" '
            f'WHERE "prcfo" IS NOT {expression}').fetchone()[0]
    finally:
        con.close()
    assert bad == 0, f'{bad} png_list row(s) whose prcfo is not its own key'


def test_the_settings_a_writer_stored_are_the_settings_two_readers_get(
        project):
    """One payload, two independent readers, same two column names.

    ``resume.read_recorded_settings`` gates a resume on these and
    ``crops.crop_settings_from_db`` rebuilds the crop geometry from them.
    They read the same table through different code, so a writer that
    renamed either column would break one of them and not the other.
    """
    from spacr.crops import crop_settings_from_db
    from spacr.resume import read_recorded_settings
    recorded = read_recorded_settings(project)
    crop = crop_settings_from_db(project)
    assert recorded, 'no settings were recorded by the writer'
    assert recorded.get('stage') == 'measure'
    # Both readers see the same keys; they differ only in how they coerce
    # the stored value.
    assert set(crop) == set(recorded)


def test_the_run_ledger_row_is_readable_by_the_module_that_reads_it(project):
    """``errors.RunLedger.stamp`` writes it; ``errors.read_run_status`` reads it.

    The INSERT is positional with no column list, so a schema drift writes
    into the wrong columns silently. Reading it back through the real reader
    is what catches that.
    """
    from spacr.errors import read_run_status
    rows = read_run_status(project)
    assert rows, 'the ledger stamped nothing readable'
    assert rows[-1]['name'] == 'measure'
    assert int(rows[-1]['n_failed']) == 0


def test_the_settings_history_keeps_every_stage_not_only_the_last(tmp_path):
    """``settings`` is replaced per stage; the history is append-only.

    The snapshot table is written with ``if_exists='replace'``, so it only
    ever holds the last stage's settings. That is by design -- but it means
    the history table is the only record that the earlier stages ran, and a
    reader asking "what was this project preprocessed with" depends on it.
    """
    from spacr.io import _save_settings_to_db, read_settings_history
    root = str(tmp_path / 'project')
    db = build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    src = os.path.join(root, 'data')
    _save_settings_to_db({'src': src, 'stage': 'preprocess', 'nchan': 2})
    _save_settings_to_db({'src': src, 'stage': 'measure', 'nchan': 3})

    history = read_settings_history(db)
    stages = {row['stage'] for row in history} if isinstance(
        history, list) else set(history['stage'])
    assert {'preprocess', 'measure'} <= stages, (
        f'the history lost a stage: {stages}')

    from spacr.resume import read_recorded_settings
    assert read_recorded_settings(db).get('stage') == 'measure', (
        'the snapshot must hold the most recent stage'
    )


def test_a_reader_that_lists_annotation_columns_sees_the_ones_added(
        fresh_project):
    """A column an annotator adds is found; the writers' columns are not.

    This is the contract between the Annotate app (which ALTERs a column
    into ``png_list``) and ``agreement``/``active_learning`` (which have to
    work out which columns those were, with nothing recording it).
    """
    from spacr.agreement import annotation_columns, load_annotations
    con = sqlite3.connect(fresh_project)
    try:
        con.execute('ALTER TABLE "png_list" ADD COLUMN "annotator_ann" INTEGER')
        con.execute('UPDATE "png_list" SET "annotator_ann" = '
                    '(CASE WHEN "cell_id" = \'o1\' THEN 1 ELSE 2 END)')
        con.commit()
    finally:
        con.close()

    assert annotation_columns(fresh_project) == ['annotator_ann']
    frame = load_annotations(fresh_project, ['annotator_ann'])
    assert len(frame) == row_count(fresh_project, 'png_list')
    # dtype=object on purpose: an INTEGER column with NULLs must not become
    # float64, or 1 becomes 1.0 and the label no longer compares equal.
    assert frame['annotator_ann'].dtype == object
    assert set(frame['annotator_ann']) == {1, 2}


def test_cell_id_means_two_types_and_a_sql_join_between_them_is_empty(project):
    """The contract that forces the migration in ``io``, measured.

    ``png_list.cell_id`` is TEXT and ``nucleus.cell_id`` is REAL. SQLite
    compares across type classes by class, and text sorts after numbers, so
    the join every naive reader would write returns nothing on a database
    where every crop has a nucleus. This is why ``io`` migrates the column
    in pandas -- via :func:`spacr.utils.object_label_from_png_id` -- instead
    of joining on it in SQL.
    """
    con = sqlite3.connect(project)
    try:
        matched = con.execute(
            'SELECT COUNT(*) FROM png_list p '
            'JOIN nucleus u ON p.cell_id = u.cell_id').fetchone()[0]
    finally:
        con.close()
    assert row_count(project, 'png_list') > 0
    assert row_count(project, 'nucleus') > 0
    assert matched == 0


# ===========================================================================
# 3. Key columns agree across the modules that touch the same tables
# ===========================================================================

def test_resume_reexports_the_schemas_field_key_rather_than_copying_it():
    """``resume`` deletes rows keyed on these; a private copy could drift."""
    from spacr import resume
    assert resume.FIELD_KEY_COLUMNS is schema.FIELD_KEY_COLUMNS


def test_predictions_private_key_copy_still_equals_the_schemas(project):
    """``spacr/predictions.py:151`` keeps its own copy of the field key.

    It is duplicated on purpose -- the module promises to import nothing
    heavy -- so the guard against drift has to be a test, and the test has
    to check it against a real database rather than against the constant it
    was copied from.
    """
    from spacr import predictions
    assert tuple(predictions._PRCFO_METADATA) == tuple(schema.FIELD_KEY_COLUMNS)
    columns = set(declared_columns(project, predictions.PNG_TABLE))
    missing = [c for c in predictions._PRCFO_METADATA if c not in columns]
    assert missing == [], (
        f'predictions rebuilds prcfo from {missing}, which png_list does not '
        f'have')


def test_predictions_names_a_table_and_key_the_writer_really_produced(project):
    from spacr import predictions
    assert predictions.PNG_TABLE in table_names(project)
    columns = set(declared_columns(project, predictions.PNG_TABLE))
    # Every key it will try in priority order must be a real column, or the
    # fallback chain silently ends at "no key at all".
    for key in predictions.KEY_PRIORITY:
        assert key in columns, f'predictions.KEY_PRIORITY names {key!r}'


def test_agreement_treats_every_writer_written_png_list_column_as_metadata(
        project):
    """A column the *writers* put in ``png_list`` is never an annotation.

    ``agreement.annotation_columns`` guesses which columns hold human
    labels. Everything ``filepaths_to_database`` wrote is metadata by
    construction, and a metadata column mistaken for an annotator is a
    kappa computed between a person and a plate coordinate.
    """
    from spacr.agreement import annotation_columns, table_columns
    written = table_columns(project)
    assert written, 'png_list has no columns'
    assert annotation_columns(project) == [], (
        f'agreement offered {annotation_columns(project)} as annotation '
        f'columns on a database with no annotations; png_list holds only '
        f'{written}')


def test_active_learning_metadata_names_exist_in_png_list(project):
    """``active_learning`` copies these across when present; names must match."""
    from spacr import active_learning
    columns = set(declared_columns(project, 'png_list'))
    # The id columns are per-crop-mode, so only the cell one is here; the
    # field key and the derived keys must all be real.
    for name in schema.FIELD_KEY_COLUMNS + ('prcfo', 'file_name', 'cell_id'):
        assert name in active_learning._METADATA_COLUMNS, (
            f'active_learning does not treat {name!r} as metadata')
        assert name in columns, f'png_list has no {name!r}'


def test_io_maps_every_crop_mode_to_the_column_its_writer_creates(tmp_path):
    """``io.PNG_LIST_ID_COLUMNS`` against what the writer actually wrote.

    Checked per crop mode by writing a project for each and reading the
    column back, rather than by comparing the map to the other map it was
    derived from.
    """
    from spacr.io import PNG_LIST_ID_COLUMNS
    from spacr.utils import PNG_OBJECT_ID_COLUMNS
    for mode, column in PNG_OBJECT_ID_COLUMNS.items():
        assert PNG_LIST_ID_COLUMNS[mode] == column
        root = str(tmp_path / mode)
        os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
        write_crops(root, 'plate1_A01_1', crop_mode=mode)
        columns = set(declared_columns(db_of(root), 'png_list'))
        assert column in columns, (mode, sorted(columns))
        others = set(PNG_OBJECT_ID_COLUMNS.values()) - {column}
        assert not (others & columns), (
            f'crop mode {mode!r} wrote more than its own id column')


def test_foreign_reserves_every_column_the_measure_writer_owns(project):
    """A foreign import must not be allowed to overwrite spaCR's own keys.

    ``foreign.RESERVED_COLUMNS`` is the list a third-party column is
    refused or renamed against. Any measure-written identity column missing
    from it can be silently replaced by an importer's column of the same
    name, after which every join in the project is against their values.
    """
    from spacr import foreign
    reserved = {str(c) for c in foreign.RESERVED_COLUMNS}
    identity = set(schema.OBJECT_TABLE_REQUIRED_COLUMNS) | {'cell_id'}
    on_disk = set(declared_columns(project, 'cell'))
    should_be_reserved = sorted((identity & on_disk) - reserved)
    assert should_be_reserved == [], (
        f'foreign.RESERVED_COLUMNS does not protect {should_be_reserved}, '
        f'which the measure writer owns in the cell table')


@pytest.mark.parametrize('table', schema.ORGANELLE_SUMMARY_TABLES)
def test_an_organelle_summary_table_keys_like_its_parent(tmp_path, table):
    """The optional owned tables obey the same key contract as the rest.

    One row per parent object, so ``object_label`` is the whole object key --
    the same shape as ``cell``. They are written only when
    ``summarize_organelles_by`` is set, which is why they are absent from
    the shared project and are built here on their own.
    """
    from spacr.utils import _merge_and_save_to_database
    root = str(tmp_path / table)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    frame = pd.DataFrame({
        'label': [1, 2, 3],
        'organelle_summary_count': [4, 0, 2],
        'organelle_summary_total_area': [40.0, 0.0, 12.0]})
    _merge_and_save_to_database(frame, pd.DataFrame(), table, root,
                                'plate1_A01_1', 'exp', False)
    db = db_of(root)

    assert table in table_names(db)
    assert table in schema.OWNED_TABLES
    columns = set(declared_columns(db, table))
    for key in schema.table_key_columns(table):
        assert key in columns, f'{table}.{key} declared by schema, absent on disk'
    assert schema.OBJECT_LABEL_KEY in columns
    assert 'cell_id' not in columns, (
        f'{table} rows are parents, not children; a cell_id would make the '
        f'row look like a child object to every reader')


def test_the_key_separator_sequencing_splits_on_is_the_one_the_writers_used(
        project):
    """``sequencing`` reads no table, but it parses spaCR's keys.

    ``spacr/sequencing.py:1027`` takes the token after the *last*
    :data:`schema.KEY_SEPARATOR` of a ``rowID`` -- count CSVs carry the
    composite ``'<plate>_<row>'`` there, and a plain ``'r1'`` has to survive
    the same call untouched. Both halves are checked against values the real
    writers put in the database, because a separator that stopped matching
    would leave the composite unsplit and silently plot one plate's rows
    under another's name.
    """
    con = sqlite3.connect(project)
    try:
        rows = [r[0] for r in con.execute('SELECT DISTINCT "rowID" FROM "cell"')]
        keys = [r[0] for r in con.execute('SELECT "prcfo" FROM "png_list"')]
        ids = [r[0] for r in con.execute('SELECT "cell_id" FROM "png_list"')]
    finally:
        con.close()
    sep = schema.KEY_SEPARATOR

    # A plain row label written by the writer must pass through unchanged.
    assert rows == ['r1']
    for value in rows:
        assert value.rsplit(sep, 1)[-1] == value

    # And the composite the count CSVs carry resolves to that same label.
    for value in rows:
        assert f'plate1{sep}{value}'.rsplit(sep, 1)[-1] == value

    # The same split is what makes prcfo's last token the object id, which
    # is the contract predictions rebuilds its merge key from.
    assert keys and ids
    assert [k.rsplit(sep, 1)[-1] for k in keys] == ids


def test_resume_discovers_exactly_the_measure_tables_it_may_delete_from(
        project):
    """Discovery is keyed on the same columns the writers wrote.

    ``discover_field_tables`` returns the tables a resume may clear. It must
    find the measure output and must not find bookkeeping tables, which
    carry no field key and whose rows a field-level delete would destroy
    wholesale.
    """
    from spacr.resume import discover_field_tables
    found = set(discover_field_tables(project))
    for table in BUILT_OBJECT_TABLES + ('png_list',):
        assert table in found, f'{table} was written per field but not found'
    for table in ('settings', 'object_counts', 'run_status',
                  'settings_history'):
        assert table not in found, (
            f'{table} is not per-field and must never be cleared by field')


# ===========================================================================
# 4. `rowid` is not a row identity in any spaCR table
# ===========================================================================

def rowid_shadowing_tables(db):
    """Every table in ``db`` that declares a column shadowing ``rowid``."""
    out = []
    for table in table_names(db):
        lowered = {c.lower() for c in declared_columns(db, table)}
        if lowered & {'rowid', 'oid', '_rowid_'}:
            out.append(table)
    return out


def test_the_writers_really_do_produce_rowid_shadowing_tables(project):
    """The precondition of the whole section, from the real writers.

    If this ever stops being true the tests below stop testing anything, so
    it is asserted rather than assumed.
    """
    shadowed = rowid_shadowing_tables(project)
    assert 'cell' in shadowed and 'png_list' in shadowed, shadowed


@pytest.mark.parametrize('table', BUILT_OBJECT_TABLES + ('png_list',))
def test_select_rowid_returns_the_plate_row_not_a_row_identity(project, table):
    """``rowid`` reads the ``rowID`` *column*, and no quoting rescues it."""
    con = sqlite3.connect(project)
    try:
        bare = [r[0] for r in con.execute(f'SELECT rowid FROM "{table}"')]
        quoted = [r[0] for r in con.execute(f'SELECT "rowid" FROM "{table}"')]
        real = [r[0] for r in con.execute(f'SELECT _rowid_ FROM "{table}"')]
    finally:
        con.close()
    assert bare, f'{table} is empty'
    assert set(bare) == {'r1'}, (
        f'SELECT rowid FROM {table} returned {sorted(set(bare))}; it is '
        f'reading the plate row column')
    assert quoted == bare, 'quoting does not reach the implicit row id'
    assert real == sorted(real) and len(set(real)) == len(real), (
        '_rowid_ is the only spelling that is a row identity here')


def test_a_delete_written_against_rowid_destroys_the_whole_table(
        fresh_project):
    """The bug, measured, on a table the real writer filled.

    Two rows carry ``object_label = 1`` (two fields). A DELETE written the
    obvious way -- select the row ids of the rows you want, delete where
    ``rowid`` is one of them -- removes every row that shares a *plate row*
    with any of them, which is the entire table.
    """
    con = sqlite3.connect(fresh_project)
    try:
        con.execute('CREATE TABLE probe AS SELECT * FROM "cell"')
        total = con.execute('SELECT COUNT(*) FROM probe').fetchone()[0]
        targeted = con.execute(
            'SELECT COUNT(*) FROM probe WHERE "object_label" = 1').fetchone()[0]
        removed = con.execute(
            'DELETE FROM probe WHERE rowid IN '
            '(SELECT s.rowid FROM probe AS s WHERE s."object_label" = 1)'
        ).rowcount
    finally:
        con.close()
    assert targeted == 2, 'two fields, one object 1 each'
    assert total == 2 * N_OBJECTS
    assert removed == total, (
        f'the rowid delete removed {removed} of {total}; if this is ever '
        f'equal to {targeted} the shadowing has been fixed and this test '
        f'should become an equality against the target count')


@pytest.mark.parametrize('table', BUILT_OBJECT_TABLES + ('png_list',))
def test_the_module_that_updates_by_row_identity_picks_a_safe_spelling(
        project, table):
    """``predictions._rowid_alias`` must return a spelling that really works.

    ``predictions`` is the one module that addresses ``png_list`` rows by
    identity -- ``UPDATE ... WHERE <alias> = ?`` -- so its choice of alias
    is load-bearing. Verified by using the alias it returns against the real
    table and checking the values are distinct row identities.
    """
    from spacr.predictions import _rowid_alias
    columns = list(declared_columns(project, table))
    alias = _rowid_alias(columns)
    assert alias.lower() not in {c.lower() for c in columns}
    con = sqlite3.connect(project)
    try:
        values = [r[0] for r in con.execute(
            f'SELECT {alias} FROM "{table}"')]
    finally:
        con.close()
    assert len(set(values)) == len(values) == row_count(project, table), (
        f'{alias} is not a row identity in {table}: {values}')
    assert all(isinstance(v, int) for v in values)


# ===========================================================================
# 5. The delete safety property, stated as the one that is actually true
# ===========================================================================

def test_the_declared_row_key_is_not_unique_so_a_keyed_delete_is_not_safe(
        fresh_project):
    """``row_key_columns`` does not identify one row. Measured.

    The writer appends, so re-measuring a field -- a resume, a re-run, an
    importer's copy sitting beside measure's rows -- puts two rows in the
    table with identical values in all five key columns. A "keyed delete"
    on them is therefore exactly as destructive as the ``rowid`` one, and
    any test asserting that a keyed delete is safe is asserting something
    false.
    """
    root = os.path.dirname(os.path.dirname(fresh_project))
    measure_field(root, 'plate1_A01_1')          # the same field, again
    keys = list(schema.object_table_schema('cell').row_key_columns())

    con = sqlite3.connect(fresh_project)
    try:
        quoted = ', '.join(f'"{k}"' for k in keys)
        rows = con.execute(f'SELECT {quoted} FROM "cell"').fetchall()
        duplicated = len(rows) - len(set(rows))
        con.execute('CREATE TABLE probe AS SELECT * FROM "cell"')
        where = ' AND '.join(f'"{k}" IS ?' for k in keys)
        removed = con.execute(f'DELETE FROM probe WHERE {where}',
                              rows[0]).rowcount
    finally:
        con.close()

    assert duplicated > 0, (
        'the writer no longer produces duplicate row keys; if it has gained '
        'a uniqueness constraint this test should assert that instead')
    assert removed == 2, (
        f'a keyed delete aimed at one object removed {removed} rows -- the '
        f'declared key does not identify a row')


@pytest.mark.parametrize('table', BUILT_OBJECT_TABLES)
def test_count_then_delete_on_one_predicate_is_the_pattern_that_holds(
        fresh_project, table):
    """The property that *is* true, stated for every object table.

    Neither of the two obvious row identities works here: ``rowid`` is a
    plate row and the declared key is not unique. What does work, and what
    ``foreign.release_canonical_copy`` is built on, is to never name a row
    at all -- count with a predicate, delete with the *same* predicate, and
    treat any difference between the two numbers as a failure rather than a
    result.

    Demonstrated on a table with duplicate row keys, which is the case both
    other approaches get wrong.
    """
    root = os.path.dirname(os.path.dirname(fresh_project))
    measure_field(root, 'plate1_A01_1', tables=(table,))   # duplicate the field

    predicate = '"fieldID" = \'f1\''
    con = sqlite3.connect(fresh_project)
    try:
        con.execute(f'CREATE TABLE probe AS SELECT * FROM "{table}"')
        counted = con.execute(
            f'SELECT COUNT(*) FROM probe WHERE {predicate}').fetchone()[0]
        removed = con.execute(
            f'DELETE FROM probe WHERE {predicate}').rowcount
        left = con.execute('SELECT COUNT(*) FROM probe').fetchone()[0]
        total = row_count(fresh_project, table)
    finally:
        con.close()

    assert counted == 2 * N_OBJECTS, 'field 1 was measured twice'
    assert removed == counted, (
        f'the delete removed {removed} where the count that gated it said '
        f'{counted}; this is the equality the release refuses on')
    assert left == total - counted, 'the other field survived untouched'


def test_a_release_with_no_import_to_release_touches_nothing(tmp_path):
    """The same pattern's zero case: no predicate, no delete, no damage.

    ``release_canonical_copy`` is called on every measure of a project that
    might have been imported into, so its behaviour on a project that was
    not is load-bearing: it must decline rather than fall through to a
    delete it cannot account for.
    """
    foreign = pytest.importorskip('spacr.foreign')
    root = str(tmp_path / 'project')
    build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    db = db_of(root)

    before = {t: row_count(db, t) for t in table_names(db)}
    assert foreign.release_canonical_copy(db, 'cell') == 0
    assert {t: row_count(db, t) for t in table_names(db)} == before, (
        'a release with no import to release changed the database')


def test_a_dry_run_release_reports_the_same_number_it_would_delete(tmp_path):
    """The count a caller is shown is the count the delete is gated on."""
    foreign = pytest.importorskip('spacr.foreign')
    root = str(tmp_path / 'project')
    build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    db = db_of(root)
    before = row_count(db, 'cell')
    assert foreign.release_canonical_copy(db, 'cell', dry_run=True) == 0
    assert row_count(db, 'cell') == before


def test_clearing_a_field_removes_only_that_fields_rows(fresh_project):
    """``resume.clear_field_rows`` is keyed on the field, and stays there.

    It is allowed to remove every row of one field -- that is its job -- but
    a second field's rows and the bookkeeping tables must be untouched.
    """
    from spacr.resume import clear_field_rows, discover_field_tables
    tables = discover_field_tables(fresh_project)
    before = {t: row_count(fresh_project, t) for t in table_names(fresh_project)}

    clear_field_rows(fresh_project, tables, 'plate1_A01_1')

    after = {t: row_count(fresh_project, t) for t in table_names(fresh_project)}
    for table in BUILT_OBJECT_TABLES + ('png_list',):
        assert after[table] == before[table] // 2, (
            f'{table}: cleared one of two fields, {before[table]} -> '
            f'{after[table]}')
    for table in ('settings', 'object_counts', 'run_status',
                  'settings_history'):
        assert after[table] == before[table], (
            f'{table} is not per-field and lost rows to a field clear')


def test_clearing_a_field_refuses_a_table_the_measure_stage_does_not_own(
        fresh_project):
    """The refusal is by name, before anything is deleted.

    ``conversion_map``, ``align_coordinates`` and ``foreign_*`` all carry
    the same four field key columns -- deliberately, so that they join to
    the measurements -- so the column test alone would have cleared them.
    """
    from spacr.resume import clear_field_rows
    con = sqlite3.connect(fresh_project)
    try:
        con.execute('CREATE TABLE conversion_map AS SELECT '
                    '"plateID","rowID","columnID","fieldID" FROM "cell"')
        con.commit()
    finally:
        con.close()
    before = row_count(fresh_project, 'conversion_map')
    assert before > 0

    with pytest.raises(ValueError) as raised:
        clear_field_rows(fresh_project, ['conversion_map'], 'plate1_A01_1')

    assert 'conversion_map' in str(raised.value)
    assert row_count(fresh_project, 'conversion_map') == before, (
        'the refusal must happen before any delete')


# ===========================================================================
# 6. Migrations: readable, or refused -- never silently misread
# ===========================================================================

def age_database(db, *, version=0, legacy_names=True):
    """Rewrite ``db`` as one an older spaCR release would have written.

    The version-1 migration is the canonicalisation of the metadata column
    spellings, so an older database is one with the legacy spellings and no
    version stamp.
    """
    con = sqlite3.connect(db)
    try:
        if legacy_names:
            renames = (('plateID', 'plate'), ('rowID', 'row'),
                       ('columnID', 'column'), ('fieldID', 'field'))
            for table in table_names(db):
                columns = set(declared_columns(db, table))
                for new, old in renames:
                    if new in columns:
                        con.execute(f'ALTER TABLE "{table}" '
                                    f'RENAME COLUMN "{new}" TO "{old}"')
        con.execute(f'PRAGMA user_version = {int(version)}')
        con.commit()
    finally:
        con.close()


def test_the_writers_stamp_a_new_database_with_the_current_version(project):
    """A database spaCR made says which schema it is."""
    con = sqlite3.connect(project)
    try:
        version = con.execute('PRAGMA user_version').fetchone()[0]
        application = con.execute('PRAGMA application_id').fetchone()[0]
    finally:
        con.close()
    assert version == CURRENT_SCHEMA_VERSION
    assert application == int.from_bytes(b'SPCR', 'big')


def test_an_older_database_is_migrated_on_read_and_its_rows_survive(tmp_path):
    """The legacy spellings are repaired in place, values intact.

    This is the "readable" half of the contract. The reader does not merely
    tolerate the old names -- it renames them, stamps the version, and
    returns the same objects it would have returned from a current database.
    """
    from spacr.io import _read_and_join_tables
    root = str(tmp_path / 'project')
    db = build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    expected = row_count(db, 'cell')
    age_database(db)
    assert 'plate' in declared_columns(db, 'cell')

    out = _read_and_join_tables(db, table_names=['cell', 'nucleus', 'png_list'])

    assert len(out) == expected
    assert out['object_label'].tolist() == list(range(1, expected + 1))
    columns = declared_columns(db, 'cell')
    for canonical in schema.FIELD_KEY_COLUMNS:
        assert canonical in columns, f'{canonical} was not restored'
    con = sqlite3.connect(db)
    try:
        assert con.execute(
            'PRAGMA user_version').fetchone()[0] == CURRENT_SCHEMA_VERSION
    finally:
        con.close()


def test_an_older_database_is_not_silently_misread_before_migration(tmp_path):
    """A legacy database read *without* migrating answers nothing, not wrongly.

    ``resume.discover_field_tables`` looks for the canonical field key
    columns. On an un-migrated database it finds none, so it returns an
    empty list -- and an empty list is a resume that does nothing, which is
    safe. The failure it must never have is finding *some* tables and
    clearing rows from them on a half-understood schema.
    """
    from spacr.resume import discover_field_tables
    root = str(tmp_path / 'project')
    db = build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    age_database(db)

    assert discover_field_tables(db) == [], (
        'a legacy database must not be partially recognised')

    from spacr.database_schema import ensure_database_schema
    ensure_database_schema(db)

    assert set(discover_field_tables(db)) >= set(BUILT_OBJECT_TABLES), (
        'after migration the same call must find the measure tables')


@pytest.mark.parametrize('reader', ['_read_db', '_read_and_join_tables'])
def test_a_newer_database_is_refused_by_name_and_left_alone(tmp_path, reader):
    """The "refused with a clear message" half, and it must not mutate.

    A database written by a future spaCR is the one case where guessing is
    unrecoverable: its columns may mean something this release does not
    know. The refusal has to name the file, both versions, and it has to
    happen before anything is written.
    """
    import spacr.io as io
    root = str(tmp_path / 'project')
    db = build_project(root, fields=('plate1_A01_1',), bookkeeping=False)
    future = CURRENT_SCHEMA_VERSION + 98
    con = sqlite3.connect(db)
    try:
        con.execute(f'PRAGMA user_version = {future}')
        con.commit()
    finally:
        con.close()
    before = declared_columns(db, 'cell')

    with pytest.raises(DatabaseSchemaTooNewError) as raised:
        if reader == '_read_db':
            io._read_db(db, ['cell'])
        else:
            io._read_and_join_tables(db, table_names=['cell'])

    message = str(raised.value)
    assert str(future) in message
    assert str(CURRENT_SCHEMA_VERSION) in message
    assert os.path.basename(db) in message
    assert declared_columns(db, 'cell') == before, (
        'the refusal must not have changed the database')
    con = sqlite3.connect(db)
    try:
        assert con.execute('PRAGMA user_version').fetchone()[0] == future
    finally:
        con.close()


def test_migrating_an_already_current_database_changes_nothing(project):
    """Idempotence: a reader may run the migration on every open."""
    from spacr.database_schema import ensure_database_schema
    before = {t: declared_columns(project, t) for t in table_names(project)}
    counts = {t: row_count(project, t) for t in table_names(project)}

    ensure_database_schema(project)

    assert {t: declared_columns(project, t)
            for t in table_names(project)} == before
    assert {t: row_count(project, t) for t in table_names(project)} == counts
