"""``cell_id`` means two types in one database, and the join has to survive it.

``png_list.cell_id`` is **TEXT** (``'o5'``) because it is the last component of
``prcfo``. Every object table's key is an **INTEGER**: ``cell.object_label``,
and ``nucleus.cell_id`` / ``pathogen.cell_id``, the child tables' link to their
parent (a REAL in practice, since ``measure`` writes NaN for "no overlapping
cell"). Two types for one identity.

At the SQL level that does not fail, it *silently matches nothing*: SQLite
compares TEXT with INTEGER by type class and text always sorts after numbers,
so ``png_list p JOIN nucleus u ON p.cell_id = u.cell_id`` returns zero rows on
a database where every crop has a nucleus. The integer is canonical --- it is
what the measurement tables key on --- so the migration happens on read, in
:func:`spacr.utils.object_label_from_png_id`, and
:func:`spacr.io._read_and_join_tables` is its one caller here.

The migration it replaces was ``series.str[1:].astype(int)``, which crashed on
four values the real writers produce routinely. Every one of them is a case
below, and every database here is built with the real writers ---
:func:`spacr.utils.filepaths_to_database` and
:func:`spacr.utils._merge_and_save_to_database`, the same two calls
``measure_crop`` makes --- because a hand-built ``png_list`` is exactly how
this stayed hidden.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.io import CropModeMismatch, _read_and_join_tables
from spacr.utils import (PNG_CROP_MODE_BY_ID_COLUMN, PNG_OBJECT_ID_COLUMNS,
                         _merge_and_save_to_database, filepaths_to_database,
                         object_label_from_png_id)


# ---------------------------------------------------------------------------
# builders -- every one of them a real spaCR writer
# ---------------------------------------------------------------------------

N_OBJECTS = 3


def measure_field(root, stem, tables=('cell', 'nucleus'), n_objects=N_OBJECTS,
                  parent_link=True):
    """Write one field's object rows through ``_merge_and_save_to_database``.

    ``parent_link=False`` reproduces ``cell_mask_dim=None``: the child table
    genuinely has no ``cell_id`` and the writer drops it from the key columns.
    """
    labels = list(range(1, n_objects + 1))
    for table in tables:
        morphology = pd.DataFrame({
            'label': labels,
            f'{table}_area': [100.0 + i for i in range(n_objects)]})
        intensity = pd.DataFrame({
            'label': labels,
            f'{table}_channel_0_mean_intensity': [5.0] * n_objects})
        if table in ('nucleus', 'pathogen') and parent_link:
            # float with NaN for "no overlapping cell", exactly as
            # measure._intensity_measurements writes it.
            morphology['cell_id'] = np.asarray(labels, dtype=float)
        _merge_and_save_to_database(morphology, intensity, table, root, stem,
                                    'exp', False)


def write_crops(root, names, crop_mode='cell'):
    """Index crop file names through the real ``filepaths_to_database``."""
    folder = os.path.join(root, 'data', f'{crop_mode}_png')
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, name) for name in names]
    filepaths_to_database(paths, {'timelapse': False}, root, crop_mode)
    return paths


def db_of(root):
    return os.path.join(root, 'measurements', 'measurements.db')


@pytest.fixture()
def project(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    return root


def crop_names(stem='plate1_A01_1', n=N_OBJECTS):
    return [f'{stem}_{i + 1}.png' for i in range(n)]


def column_type(db, table, column):
    con = sqlite3.connect(db)
    try:
        return {r[1]: r[2] for r in con.execute(f'PRAGMA table_info("{table}")')}[column]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. the contract itself, on a database the real writers made
# ---------------------------------------------------------------------------

def test_png_list_and_nucleus_declare_cell_id_as_two_different_types(project):
    """The finding, stated as a fact about the schema rather than a guess."""
    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names())
    db = db_of(project)

    assert column_type(db, 'png_list', 'cell_id') == 'TEXT'
    assert column_type(db, 'nucleus', 'cell_id') == 'REAL'

    con = sqlite3.connect(db)
    try:
        stored = [r[0] for r in con.execute('SELECT cell_id FROM png_list')]
        matched = con.execute(
            'SELECT COUNT(*) FROM png_list p '
            'JOIN nucleus u ON p.cell_id = u.cell_id').fetchone()[0]
        crops, nuclei = (con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                         for t in ('png_list', 'nucleus'))
    finally:
        con.close()

    assert stored == ['o1', 'o2', 'o3']
    assert (crops, nuclei) == (3, 3)
    # Three crops, three nuclei, one per cell -- and a SQL join on the shared
    # column name matches nothing at all. This is why the join is done in
    # pandas after an explicit migration, not in SQL on the raw column.
    assert matched == 0


def test_the_migration_lands_png_list_on_the_object_tables_integer(project):
    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names())

    out = _read_and_join_tables(db_of(project),
                               table_names=['cell', 'nucleus', 'png_list'])

    assert out['object_label'].tolist() == [1, 2, 3]
    assert pd.api.types.is_integer_dtype(out['object_label'])
    assert out['png_path'].notna().all()
    # and the nucleus roll-up, whose key is the REAL cell_id, still lands
    assert out['count_nucleus'].tolist() == [1, 1, 1]


# ---------------------------------------------------------------------------
# 2. the four values .str[1:].astype(int) died on
# ---------------------------------------------------------------------------

def test_a_crop_over_several_cells_or_none_no_longer_kills_the_read(project, capsys):
    """``_generate_names`` writes ``..._multi.png`` / ``..._none.png``.

    A crop whose bounding box covers several cells, or none, is an ordinary
    outcome of a real segmentation --- and ``'omulti'``/``'onone'`` used to
    raise ``ValueError: invalid literal for int() with base 10: 'multi'`` and
    take the whole database read down with them.
    """
    measure_field(project, 'plate1_A01_1')
    write_crops(project, ['plate1_A01_1_1.png', 'plate1_A01_1_multi.png',
                          'plate1_A01_1_none.png'])

    out = _read_and_join_tables(db_of(project),
                               table_names=['cell', 'nucleus', 'png_list'])

    assert len(out) == 3                       # every cell still measured
    assert out['png_path'].notna().sum() == 1  # only object 1 has a real crop
    message = capsys.readouterr().out
    assert 'omulti' in message and 'onone' in message
    assert 'skipped' in message


def test_an_unparseable_crop_name_is_reported_not_mangled(project, capsys):
    """``_map_wells_png`` writes ``'error'``; ``.str[1:]`` made it ``'rror'``.

    The old failure did not even name the value it choked on.
    """
    measure_field(project, 'plate1_A01_1')
    write_crops(project, ['plate1_A01_1_1.png', 'rubbish.png'])

    out = _read_and_join_tables(db_of(project),
                               table_names=['cell', 'png_list'])

    assert len(out) == 3
    assert out['png_path'].notna().sum() == 1
    assert "'error'" in capsys.readouterr().out


def test_two_crop_modes_in_one_png_list_no_longer_crash(project, capsys):
    """``crop_mode=['cell','nucleus']`` writes both id columns into one table.

    Each row has exactly one of them; the other is NULL. The NULLs used to
    reach ``astype(int)`` as ``None`` --- ``TypeError: int() argument must be a
    string, a bytes-like object or a real number, not 'NoneType'``.
    """
    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names(), crop_mode='cell')
    write_crops(project, crop_names(), crop_mode='nucleus')
    db = db_of(project)

    con = sqlite3.connect(db)
    try:
        columns = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
        rows = con.execute('SELECT COUNT(*) FROM png_list').fetchone()[0]
    finally:
        con.close()
    assert {'cell_id', 'nucleus_id'} <= set(columns)
    assert rows == 6

    out = _read_and_join_tables(db, table_names=['cell', 'nucleus', 'png_list'])

    assert len(out) == 3
    # the three cell crops attach; the three nucleus crops are another mode's
    assert out['png_path'].notna().sum() == 3
    assert out['png_path'].str.contains('cell_png').all()
    printed = capsys.readouterr().out
    assert '3 of 6 rows have no cell_id' in printed


def test_png_list_of_another_crop_mode_is_refused_by_name(project):
    """Nucleus crops only: there is no ``cell_id`` column at all.

    ``KeyError: "['cell_id'] not in index"`` named neither the table nor the
    setting behind it.
    """
    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names(), crop_mode='nucleus')

    with pytest.raises(CropModeMismatch) as excinfo:
        _read_and_join_tables(db_of(project),
                             table_names=['cell', 'nucleus', 'png_list'])

    message = str(excinfo.value)
    assert 'cell_id' in message
    assert 'nucleus' in message and 'nucleus_id' in message
    assert 'crop_mode' in message


def test_a_child_table_with_no_parent_link_is_left_out_not_fatal(project, capsys):
    """``cell_mask_dim=None`` -> the child table has no ``cell_id`` column.

    ``_merge_and_save_to_database`` supports that case deliberately. The
    roll-up onto the cell then has no key, and used to be a bare
    ``KeyError('cell_id')``.
    """
    measure_field(project, 'plate1_A01_1', parent_link=False)
    write_crops(project, crop_names())

    out = _read_and_join_tables(db_of(project),
                               table_names=['cell', 'nucleus', 'png_list'])

    assert len(out) == 3
    assert out['png_path'].notna().all()
    assert 'count_nucleus' not in out.columns
    assert 'without a cell mask' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 3. the migration function on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    ('o1', 1.0),
    ('o42', 42.0),
    ('O7', 7.0),          # SQLite identifiers are case-insensitive; be liberal
    ('o007', 7.0),        # zero padding must not make a second object
    (' o8 ', 8.0),
    ('7', 7.0),           # a bare label, from a database migrated elsewhere
    (7, 7.0),             # already INTEGER
    (7.0, 7.0),           # REAL affinity
    ('omulti', None),
    ('onone', None),
    ('error', None),
    (None, None),
    (float('nan'), None),
    (7.5, None),          # a fractional label is not a label
    (True, None),         # True is not object 1
])
def test_object_label_from_png_id(value, expected):
    got = object_label_from_png_id(pd.Series([value], dtype=object)).iloc[0]
    if expected is None:
        assert pd.isna(got)
    else:
        assert got == expected


def test_object_label_from_png_id_on_an_empty_column():
    out = object_label_from_png_id(pd.Series([], dtype=object))
    assert len(out) == 0
    assert out.dtype == float


def test_the_crop_mode_map_is_a_bijection_and_matches_the_writer(tmp_path):
    """The mapping the reader trusts is the one the writer used.

    Checked against real output rather than against itself: each crop mode is
    written into its own project and the column that appears is the one the
    map names.
    """
    assert PNG_CROP_MODE_BY_ID_COLUMN == {v: k for k, v
                                          in PNG_OBJECT_ID_COLUMNS.items()}
    for mode, column in PNG_OBJECT_ID_COLUMNS.items():
        root = str(tmp_path / mode)
        os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
        write_crops(root, crop_names(), crop_mode=mode)
        con = sqlite3.connect(db_of(root))
        try:
            columns = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
        finally:
            con.close()
        assert column in columns, (mode, columns)
        others = set(PNG_OBJECT_ID_COLUMNS.values()) - {column}
        assert not (others & set(columns))


# ---------------------------------------------------------------------------
# 4. the other reader of the same contract
# ---------------------------------------------------------------------------

def test_read_and_merge_data_drops_the_other_modes_rows_deliberately(project,
                                                                    capsys):
    """``_read_and_merge_data`` rebuilds ``prcfo`` as ``prcf + '_' + cell_id``.

    A NULL ``cell_id`` made that ``'<field>_None'`` --- a key that matched
    nothing, so the right answer came out for the wrong reason and would have
    stopped coming out the day an object was called ``None``.
    """
    from spacr.io import _read_and_merge_data

    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names(), crop_mode='cell')
    write_crops(project, crop_names(), crop_mode='nucleus')

    merged = _read_and_merge_data([db_of(project)],
                                  ['cell', 'nucleus', 'png_list'],
                                  verbose=False)
    frame = merged[0] if isinstance(merged, tuple) else merged

    assert len(frame) == 3
    assert frame['png_path'].notna().all()
    assert frame['png_path'].str.contains('cell_png').all()
    assert '3 of 6 rows are not usable cell crops' in capsys.readouterr().out


def test_read_and_merge_data_refuses_a_png_list_of_another_crop_mode(project):
    from spacr.io import _read_and_merge_data

    measure_field(project, 'plate1_A01_1')
    write_crops(project, crop_names(), crop_mode='nucleus')

    with pytest.raises(CropModeMismatch) as excinfo:
        _read_and_merge_data([db_of(project)],
                             ['cell', 'nucleus', 'png_list'], verbose=False)
    assert 'nucleus_id' in str(excinfo.value)
