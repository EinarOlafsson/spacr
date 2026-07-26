"""Coverage for the SQLite / array-merging helpers in ``spacr.io``.

Targets ``_read_and_join_tables``, ``_save_settings_to_db``,
``_save_mask_timelapse_as_gif``, ``_save_object_counts_to_database``,
``_create_database``, ``save_object_mask``, ``_load_and_concatenate_arrays``
and ``_results_to_csv`` — in particular the branches that only fire when a
table is missing, a mask has no background label, tracks are overlaid on a
timelapse, or pathogen/organelle mask folders take part in the merge.

Everything here is CPU-only, offline and writes exclusively into ``tmp_path``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# _read_and_join_tables
# ---------------------------------------------------------------------------

def _write_png_list(db_path, n=3):
    """png_list rows with the 'o<N>' cell_id strings spacr writes."""
    png = pd.DataFrame({
        'cell_id': [f'o{i}' for i in range(1, n + 1)],
        'png_path': [f'/data/plate1_A01_1_{i}.png' for i in range(1, n + 1)],
        'plateID': ['plate1'] * n,
        'rowID': ['r01'] * n,
        'columnID': ['c01'] * n,
        'fieldID': ['1'] * n,
    })
    con = sqlite3.connect(str(db_path))
    try:
        png.to_sql('png_list', con, index=False)
    finally:
        con.close()
    return png


def test_read_and_join_tables_returns_png_list_when_cell_table_missing(
        tmp_path, capsys):
    """png_list present but no cell table -> the png_list frame is returned.

    Exercises the ``else: print(...); return png_list_df`` escape hatch: the
    'o<N>' cell_id strings must come back as an integer ``object_label``.
    """
    from spacr.io import _read_and_join_tables

    db = tmp_path / 'measurements.db'
    _write_png_list(db, n=3)

    out = _read_and_join_tables(str(db))

    captured = capsys.readouterr().out
    assert 'Cell table not found in database tables.' in captured

    assert isinstance(out, pd.DataFrame)
    # cell_id was renamed and converted, the raw column is gone.
    assert 'cell_id' not in out.columns
    assert list(out.columns) == ['object_label', 'png_path', 'plateID',
                                 'rowID', 'columnID', 'fieldID']
    assert out['object_label'].tolist() == [1, 2, 3]
    assert np.issubdtype(out['object_label'].dtype, np.integer)
    # png paths are carried through verbatim.
    assert out['png_path'].iloc[0].endswith('plate1_A01_1_1.png')


def test_read_and_join_tables_joins_cell_png_list_and_aggregates_nucleus(
        tmp_path):
    """The happy path: cell + png_list merge, nucleus rows collapse per cell."""
    from spacr.io import _read_and_join_tables

    db = tmp_path / 'measurements.db'
    _write_png_list(db, n=2)

    cell = pd.DataFrame({
        'object_label': [1, 2],
        'prcf': ['plate1_r01_c01_1', 'plate1_r01_c01_1'],
        'plateID': ['plate1', 'plate1'],
        'rowID': ['r01', 'r01'],
        'columnID': ['c01', 'c01'],
        'fieldID': ['1', '1'],
        'cell_area': [100.0, 200.0],
    })
    # Two nuclei in cell 1, one in cell 2 -> count_nucleus 2 / 1.
    nucleus = pd.DataFrame({
        'cell_id': [1, 1, 2],
        'prcf': ['plate1_r01_c01_1'] * 3,
        'nucleus_area': [10.0, 20.0, 50.0],
    })
    con = sqlite3.connect(str(db))
    try:
        cell.to_sql('cell', con, index=False)
        nucleus.to_sql('nucleus', con, index=False)
    finally:
        con.close()

    out = _read_and_join_tables(str(db))

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    # png_path came from the png_list merge.
    assert out.loc[out['object_label'] == 1, 'png_path'].iloc[0].endswith(
        'plate1_A01_1_1.png')
    # nucleus rows were aggregated (mean area) and counted.
    assert 'count_nucleus' in out.columns
    row1 = out[out['object_label'] == 1].iloc[0]
    assert row1['count_nucleus'] == 2
    assert row1['nucleus_area'] == pytest.approx(15.0)
    row2 = out[out['object_label'] == 2].iloc[0]
    assert row2['count_nucleus'] == 1
    assert row2['nucleus_area'] == pytest.approx(50.0)


def test_read_and_join_tables_merges_cytoplasm_with_suffix(tmp_path):
    """A cytoplasm table is left-joined on (object_label, prcf).

    Overlapping column names must gain the ``_cytoplasm`` suffix while the
    cell-table column keeps its bare name.
    """
    from spacr.io import _read_and_join_tables

    db = tmp_path / 'measurements.db'
    cell = pd.DataFrame({
        'object_label': [1, 2],
        'prcf': ['plate1_r01_c01_1'] * 2,
        'area': [100.0, 200.0],
    })
    cytoplasm = pd.DataFrame({
        'object_label': [1],           # cell 2 has no cytoplasm row
        'prcf': ['plate1_r01_c01_1'],
        'area': [70.0],
    })
    con = sqlite3.connect(str(db))
    try:
        cell.to_sql('cell', con, index=False)
        cytoplasm.to_sql('cytoplasm', con, index=False)
    finally:
        con.close()

    out = _read_and_join_tables(str(db), table_names=['cell', 'cytoplasm'])

    assert len(out) == 2
    assert 'area_cytoplasm' in out.columns
    row1 = out[out['object_label'] == 1].iloc[0]
    assert row1['area'] == pytest.approx(100.0)
    assert row1['area_cytoplasm'] == pytest.approx(70.0)
    # left join -> the unmatched cell keeps NaN for the cytoplasm column.
    assert np.isnan(out[out['object_label'] == 2].iloc[0]['area_cytoplasm'])


def test_read_and_join_tables_missing_tables_are_reported_and_skipped(
        tmp_path, capsys):
    """A requested table that does not exist is announced, not fatal."""
    from spacr.io import _read_and_join_tables

    db = tmp_path / 'measurements.db'
    con = sqlite3.connect(str(db))
    try:
        pd.DataFrame({'object_label': [1], 'prcf': ['p_r_c_1'],
                      'cell_area': [3.0]}).to_sql('cell', con, index=False)
    finally:
        con.close()

    out = _read_and_join_tables(str(db), table_names=['cell', 'cytoplasm'])

    assert 'Table cytoplasm not found in the database.' in capsys.readouterr().out
    # Only the cell table survived, unchanged.
    assert list(out['object_label']) == [1]
    assert out['cell_area'].iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _save_settings_to_db
# ---------------------------------------------------------------------------

def test_save_settings_to_db_writes_stringified_settings(tmp_path):
    """Every value lands in the settings table as its ``str()`` form."""
    from spacr.io import _save_settings_to_db

    src = tmp_path / 'plate1' / 'images'
    src.mkdir(parents=True)
    settings = {'src': str(src), 'diameter': 30, 'plot': False,
                'channels': [0, 1]}

    _save_settings_to_db(settings)

    db = tmp_path / 'plate1' / 'measurements' / 'measurements.db'
    assert db.is_file()
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql('SELECT * FROM settings', con)
    finally:
        con.close()
    mapping = dict(zip(df['setting_key'], df['setting_value']))
    assert mapping['diameter'] == '30'
    assert mapping['plot'] == 'False'
    assert mapping['channels'] == '[0, 1]'
    assert mapping['src'] == str(src)


def test_save_settings_to_db_replaces_previous_table(tmp_path):
    """A second save replaces the table rather than appending to it."""
    from spacr.io import _save_settings_to_db

    src = tmp_path / 'plate1' / 'images'
    src.mkdir(parents=True)
    _save_settings_to_db({'src': str(src), 'diameter': 30})
    _save_settings_to_db({'src': str(src), 'diameter': 45})

    db = tmp_path / 'plate1' / 'measurements' / 'measurements.db'
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql('SELECT * FROM settings', con)
    finally:
        con.close()
    assert len(df) == 2  # src + diameter only, not 4 rows
    assert dict(zip(df['setting_key'], df['setting_value']))['diameter'] == '45'


# ---------------------------------------------------------------------------
# _save_mask_timelapse_as_gif
# ---------------------------------------------------------------------------

def test_save_mask_timelapse_as_gif_overlays_tracks(tmp_path):
    """A non-None tracks_df draws one white polyline per track_id.

    The track-overlay branch inside ``_update`` is only reachable when
    ``tracks_df`` is not None, so this drives the animation with a real
    two-track frame table and asserts the GIF was written.
    """
    import matplotlib.colors as mcolors
    from PIL import Image
    from spacr.io import _save_mask_timelapse_as_gif

    masks = []
    for t in range(2):
        m = np.zeros((12, 12), np.uint16)
        m[2 + t:5 + t, 2:5] = 1
        m[7:10, 7:10] = 2
        masks.append(m)

    tracks_df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'x': [3.0, 4.0, 8.0, 8.0],
        'y': [3.0, 4.0, 8.0, 9.0],
    })
    out = tmp_path / 'tracks.gif'

    _save_mask_timelapse_as_gif(masks, tracks_df, str(out), cmap='viridis',
                                norm=mcolors.Normalize(vmin=0, vmax=2),
                                filenames=['frame0.npy', 'frame1.npy'])

    assert out.is_file() and out.stat().st_size > 0
    with Image.open(str(out)) as im:
        assert im.format == 'GIF'
        assert im.n_frames == len(masks)


# ---------------------------------------------------------------------------
# _save_object_counts_to_database
# ---------------------------------------------------------------------------

def _read_counts(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql('SELECT * FROM object_counts', con)
    finally:
        con.close()


def test_save_object_counts_counts_masks_without_background(tmp_path):
    """A mask whose smallest label is not 0 counts *every* unique label.

    ``_count_objects`` subtracts one only when label 0 is present; a fully
    tiled mask (no background at all) must therefore report len(unique).
    """
    from spacr.io import _save_object_counts_to_database

    no_background = np.array([[1, 1, 2], [2, 3, 3], [3, 1, 2]], dtype=np.uint16)
    with_background = np.array([[0, 0, 1], [1, 2, 0], [0, 0, 2]], dtype=np.uint16)
    db = tmp_path / 'counts.db'

    _save_object_counts_to_database(
        [no_background, with_background], 'cell',
        ['full.npy', 'sparse.npy'], str(db), '_before_filtration')

    df = _read_counts(db).set_index('file_name')
    assert set(df['count_type']) == {'cell_before_filtration'}
    # 3 labels, no background -> 3 (not 2).
    assert int(df.loc['full.npy', 'object_count']) == 3
    # 0,1,2 -> background dropped -> 2.
    assert int(df.loc['sparse.npy', 'object_count']) == 2


def test_save_object_counts_upserts_on_conflict(tmp_path):
    """Re-saving the same (file_name, count_type) updates instead of duplicating."""
    from spacr.io import _save_object_counts_to_database

    db = tmp_path / 'counts.db'
    first = np.array([[0, 1], [1, 0]], dtype=np.uint16)          # 1 object
    second = np.array([[0, 1], [2, 3]], dtype=np.uint16)         # 3 objects

    _save_object_counts_to_database([first], 'cell', ['f.npy'], str(db), '')
    assert int(_read_counts(db)['object_count'].iloc[0]) == 1

    _save_object_counts_to_database([second], 'cell', ['f.npy'], str(db), '')
    df = _read_counts(db)
    assert len(df) == 1                       # upsert, not a new row
    assert int(df['object_count'].iloc[0]) == 3
    assert df['count_type'].iloc[0] == 'cell'


def test_save_object_counts_empty_mask_reports_zero(tmp_path):
    """An all-background mask records a count of 0 rather than raising."""
    from spacr.io import _save_object_counts_to_database

    db = tmp_path / 'counts.db'
    _save_object_counts_to_database([np.zeros((4, 4), np.uint16)], 'pathogen',
                                    ['empty.npy'], str(db), '_after')
    df = _read_counts(db)
    assert int(df['object_count'].iloc[0]) == 0
    assert df['count_type'].iloc[0] == 'pathogen_after'


# ---------------------------------------------------------------------------
# _create_database
# ---------------------------------------------------------------------------

def test_create_database_creates_openable_file(tmp_path):
    from spacr.io import _create_database

    db = tmp_path / 'new.db'
    _create_database(str(db))
    assert db.is_file()
    con = sqlite3.connect(str(db))
    try:
        con.execute('CREATE TABLE t (a INTEGER)')
        con.execute('INSERT INTO t VALUES (1)')
        assert con.execute('SELECT a FROM t').fetchall() == [(1,)]
    finally:
        con.close()


def test_create_database_swallows_connection_errors(tmp_path, capsys):
    """A path sqlite cannot open is reported, not raised."""
    from spacr.io import _create_database

    bad = tmp_path / 'missing_dir' / 'nested.db'   # parent does not exist
    _create_database(str(bad))                     # must not raise
    assert not bad.exists()
    assert 'unable to open database file' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# save_object_mask
# ---------------------------------------------------------------------------

def test_save_object_mask_uncompressed_preserves_labels(tmp_path):
    """compression='none' maps to no codec and labels survive verbatim."""
    from spacr.io import save_object_mask, _load_array_any

    mask = np.zeros((8, 8), np.int32)
    mask[1:4, 1:4] = 9
    mask[5:7, 5:7] = 65535
    out = save_object_mask(str(tmp_path), 'plate1_A01_f1.npy', mask,
                           compression='none')
    assert out == str(tmp_path / 'plate1_A01_f1.tif')
    back = _load_array_any(out)
    assert back.dtype == np.uint16
    assert np.array_equal(back, mask.astype(np.uint16))


# ---------------------------------------------------------------------------
# _load_and_concatenate_arrays
# ---------------------------------------------------------------------------

def _build_merge_src(root, mask_names, n_channels=2, shape=(10, 10)):
    """A stack/ folder plus one mask folder per name in ``mask_names``."""
    h, w = shape
    stack_dir = os.path.join(root, 'stack')
    os.makedirs(stack_dir)
    img = np.zeros((h, w, n_channels), np.float32)
    for c in range(n_channels):
        img[..., c] = float(c + 1)
    np.save(os.path.join(stack_dir, 'fov.npy'), img)

    masks = {}
    for i, name in enumerate(mask_names):
        m = np.zeros((h, w), np.uint16)
        m[1 + i:4 + i, 1:4] = 10 * (i + 1)
        d = os.path.join(root, 'masks', name)
        os.makedirs(d)
        np.save(os.path.join(d, 'fov.npy'), m)
        masks[name] = m
    return img, masks


def test_load_and_concatenate_arrays_includes_pathogen_and_organelle_dims(
        tmp_path):
    """All four mask folders are appended when their channel dims are given.

    The pathogen and organelle folder-path branches only run when
    ``pathogen_chann_dim`` / ``organelle_chann_dim`` are not None (or the
    folder exists), so this drives them and checks each mask ends up as its
    own slice of the merged array, in folder order.
    """
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    names = ['cell_mask_stack', 'nucleus_mask_stack',
             'pathogen_mask_stack', 'organelle_mask_stack']
    img, masks = _build_merge_src(root, names, n_channels=2)

    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                 nucleus_chann_dim=1, pathogen_chann_dim=2,
                                 organelle_chann_dim=3)

    merged = np.load(os.path.join(root, 'merged', 'fov.npy'))
    # 2 image channels + 4 masks.
    assert merged.shape == (10, 10, 6)
    assert np.array_equal(merged[..., 0], img[..., 0])
    assert np.array_equal(merged[..., 1], img[..., 1])
    for offset, name in enumerate(names):
        assert np.array_equal(merged[..., 2 + offset], masks[name]), name


def test_load_and_concatenate_arrays_picks_up_folders_when_dims_are_none(
        tmp_path):
    """pathogen/organelle folders are merged on existence alone.

    With every ``*_chann_dim`` None the folders must still be discovered via
    the ``os.path.exists`` half of each condition.
    """
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    names = ['pathogen_mask_stack', 'organelle_mask_stack']
    img, masks = _build_merge_src(root, names, n_channels=1)

    _load_and_concatenate_arrays(root, channels=None, cell_chann_dim=None,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)

    merged = np.load(os.path.join(root, 'merged', 'fov.npy'))
    assert merged.shape == (10, 10, 3)          # 1 image channel + 2 masks
    assert np.array_equal(merged[..., 0], img[..., 0])
    assert np.array_equal(merged[..., 1], masks['pathogen_mask_stack'])
    assert np.array_equal(merged[..., 2], masks['organelle_mask_stack'])


def test_load_and_concatenate_arrays_skips_files_missing_from_a_mask_folder(
        tmp_path):
    """A stack file absent from one mask folder produces no merged output."""
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    _build_merge_src(root, ['cell_mask_stack'], n_channels=2)
    # A second FOV that only exists in stack/.
    np.save(os.path.join(root, 'stack', 'lonely.npy'),
            np.zeros((10, 10, 2), np.float32))

    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)

    merged_dir = os.path.join(root, 'merged')
    assert sorted(os.listdir(merged_dir)) == ['fov.npy']


def test_load_and_concatenate_arrays_ignores_non_npy_reference_files(tmp_path):
    """Non-.npy entries in stack/ are counted but never merged."""
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    _build_merge_src(root, ['cell_mask_stack'], n_channels=2)
    with open(os.path.join(root, 'stack', 'notes.txt'), 'w') as fh:
        fh.write('not an array')

    _load_and_concatenate_arrays(root, channels=[0, 1], cell_chann_dim=0,
                                 nucleus_chann_dim=None,
                                 pathogen_chann_dim=None,
                                 organelle_chann_dim=None)

    assert os.listdir(os.path.join(root, 'merged')) == ['fov.npy']


# ---------------------------------------------------------------------------
# _results_to_csv
# ---------------------------------------------------------------------------

def test_results_to_csv_writes_both_frames_with_index(tmp_path):
    """cells.csv / wells.csv are written under results/ and returned as-is."""
    from spacr.io import _results_to_csv

    cells = pd.DataFrame({'cell_area': [1.0, 2.0]}, index=['o1', 'o2'])
    wells = pd.DataFrame({'mean_area': [1.5]}, index=['plate1_A01'])

    out_cells, out_wells = _results_to_csv(str(tmp_path), cells, wells)

    assert out_cells is cells and out_wells is wells
    cells_csv = tmp_path / 'results' / 'cells.csv'
    wells_csv = tmp_path / 'results' / 'wells.csv'
    assert cells_csv.is_file() and wells_csv.is_file()

    back = pd.read_csv(cells_csv, index_col=0)
    assert list(back.index) == ['o1', 'o2']
    assert back['cell_area'].tolist() == [1.0, 2.0]
    back_wells = pd.read_csv(wells_csv, index_col=0)
    assert list(back_wells.index) == ['plate1_A01']
    assert back_wells['mean_area'].tolist() == [1.5]
