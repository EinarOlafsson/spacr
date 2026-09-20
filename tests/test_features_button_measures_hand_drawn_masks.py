"""A table of hand-picked files measures to the same place Measure does.

Instruction 421. The claim the FEATURES button makes is not "it produces
measurements" -- it is "it produces the MEASURE MODULE's measurements", and
the only way to check that is to run both on the same pixels and compare what
they wrote. :func:`test_a_table_run_and_a_measure_run_agree` does exactly
that: the same two fields go through
:func:`spacr.measure.measure_from_field_table` and through
:func:`spacr.measure.measure_crop` on a hand-built ``merged/`` folder, and
the folder trees, the database tables, the columns and the measured numbers
are compared.

Qt-free. The window that drives this table is tested in
``tests/qt/test_features_button_opens_the_measure_table.py``.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest
import tifffile

from spacr.crops import MERGED_LAYOUT_SIDECAR
from spacr.errors import ConfigurationError
from spacr.measure import (
    FieldRow,
    FieldTable,
    assign_paths_by_regex,
    field_table_settings,
    mask_role_of,
    measure_from_field_table,
    write_field_table_project,
)

#: The pattern the FEATURES box offers, spelled out here so a change to the
#: default is a change to a test rather than a silent change to behaviour.
REGEX = r'(?P<field>fov\d+)_(?:C(?P<channel>\d+)|(?P<mask>cell|nucleus)_mask)'

#: What the run is asked for. Every expensive per-object measurement is off:
#: this test is about WHERE the numbers go and whether the two routes agree,
#: not about the numbers themselves, and the full set takes minutes.
LEAN = {
    "n_jobs": 1, "save_png": True, "plot": False, "verbose": False,
    "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
    "cytoplasm_min_size": 0, "homogeneity": False, "radial_dist": False,
    "spatial_measurements": False, "object_distances": False,
    "calculate_correlation": False,
}


def _write(path, array):
    """Write one 16-bit TIFF, making its folder first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


def _planes(field):
    """The two channels and two masks of one synthetic field."""
    yy, xx = np.indices((48, 48))
    c1 = ((yy * 31 + xx * field) % 4096).astype(np.uint16)
    c2 = ((xx * 17 + yy * 3 * field) % 4096).astype(np.uint16)
    cell = np.zeros((48, 48), np.uint16)
    cell[4:44, 4:44] = 1
    nucleus = np.zeros((48, 48), np.uint16)
    nucleus[14:30, 15:31] = 1
    return c1, c2, cell, nucleus


@pytest.fixture
def drawn(tmp_path):
    """Two fields as a user leaves them: four loose files each."""
    folder = tmp_path / "drawn"
    for field in (1, 2):
        c1, c2, cell, nucleus = _planes(field)
        _write(str(folder / f"fov00{field}_C1.tif"), c1)
        _write(str(folder / f"fov00{field}_C2.tif"), c2)
        _write(str(folder / f"fov00{field}_cell_mask.tif"), cell)
        _write(str(folder / f"fov00{field}_nucleus_mask.tif"), nucleus)
    _write(str(folder / "notes_readme.tif"), np.zeros((4, 4), np.uint16))
    return folder


def _paths(folder):
    """Every file in ``folder``, as a drop would hand them over."""
    return [str(folder / name) for name in sorted(os.listdir(folder))]


def _tree(root):
    """Every file under ``root``, relative and sorted, minus the volatile ones.

    The SQLite side files and the run's own artifact database come and go
    with WAL checkpointing and with timing, so comparing them compares the
    machine rather than the two routes.
    """
    found = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(("-wal", "-shm")) or name == "artifacts.db":
                continue
            found.add(os.path.relpath(
                os.path.join(dirpath, name), root))
    return found


def _tables(db_path):
    """``table -> (row count, column names)`` for every table in ``db_path``."""
    out = {}
    with sqlite3.connect(db_path) as connection:
        names = [row[0] for row in connection.execute(
            "select name from sqlite_master where type='table'")]
        for name in names:
            count = connection.execute(
                f'select count(*) from "{name}"').fetchone()[0]
            columns = tuple(sorted(
                row[1] for row in connection.execute(
                    f'pragma table_info("{name}")')))
            out[name] = (count, columns)
    return out


def test_one_regex_fills_the_table_and_says_what_it_could_not_place(drawn):
    """The table pivots loose files into fields by channel and by object."""
    found = assign_paths_by_regex(_paths(drawn), REGEX)

    assert found.table.n_channels == 2
    assert found.table.ordered_roles() == ('cell', 'nucleus')
    assert [row.label for row in found.table.rows] == ['fov001', 'fov002']
    assert found.table.mask_dims() == {'cell': 2, 'nucleus': 3}
    assert found.table.problems() == []
    assert len(found.assigned) == 8

    left_over = [os.path.basename(path) for path, _why in found.unassigned]
    assert left_over == ['notes_readme.tif']
    assert found.unassigned[0][1]


def test_channel_tokens_are_ranked_so_c1_is_channel_zero():
    """``C1``/``C2`` and ``w2``/``w7`` both start at channel 0.

    The decision recorded in :func:`spacr.measure.assign_paths_by_regex`: a
    literal read of ``C1`` leaves channel 0 empty for ever, so the distinct
    tokens are ranked instead. ``C10`` sorting after ``C9`` is the part a
    plain string sort gets wrong.
    """
    pattern = r'(?P<field>f\d+)_(?P<channel>[A-Za-z]*\d+)'
    found = assign_paths_by_regex(
        ['/d/f1_C1.tif', '/d/f1_C2.tif', '/d/f1_C10.tif'], pattern)

    assert found.table.n_channels == 3
    row = found.table.rows[0]
    assert os.path.basename(row.channels[0]) == 'f1_C1.tif'
    assert os.path.basename(row.channels[1]) == 'f1_C2.tif'
    assert os.path.basename(row.channels[2]) == 'f1_C10.tif'

    other = assign_paths_by_regex(
        ['/d/f1_w2.tif', '/d/f1_w7.tif'], pattern)
    assert other.table.rows[0].channels[0].endswith('w2.tif')
    assert other.table.rows[0].channels[1].endswith('w7.tif')


def test_a_mask_column_takes_the_names_a_laboratory_uses():
    """``nuclei``, ``parasite`` and ``Organelle 2`` name roles spaCR has."""
    assert mask_role_of('nuclei') == 'nucleus'
    assert mask_role_of('Parasite') == 'pathogen'
    assert mask_role_of('organelle2') == 'organelleb'
    assert mask_role_of('mito') == 'organelle'
    assert mask_role_of('spleen') is None


def test_an_incomplete_table_names_its_gaps_and_writes_nothing(tmp_path,
                                                               drawn):
    """A missing file is a sentence, not a half-written project."""
    found = assign_paths_by_regex(_paths(drawn), REGEX)
    del found.table.rows[0].channels[1]
    del found.table.rows[1].masks['nucleus']

    problems = found.table.problems()
    assert any('channel 2' in text and 'fov001' in text for text in problems)
    assert any('nucleus mask' in text and 'fov002' in text
               for text in problems)
    assert not found.table.is_ready()

    destination = tmp_path / "refused"
    with pytest.raises(ConfigurationError):
        write_field_table_project(found.table, str(destination))
    assert not os.path.isdir(destination / "merged")


def test_two_rows_claiming_one_field_is_refused(drawn):
    """One row would overwrite the other's merged array; say so first."""
    found = assign_paths_by_regex(_paths(drawn), REGEX)
    found.table.rows[1].field = found.table.rows[0].field

    assert any('would overwrite' in text
               for text in found.table.problems())


def test_the_table_decides_the_metadata_settings_and_nothing_else(drawn):
    """Channel dims and mask dims come from the table; the rest is the user's."""
    found = assign_paths_by_regex(_paths(drawn), REGEX)

    resolved = field_table_settings(
        found.table, {"save_png": False, "png_size": [64, 64]},
        dst="/tmp/project")

    assert resolved['src'] == os.path.join('/tmp/project', 'merged')
    assert resolved['channels'] == [0, 1]
    assert resolved['cell_mask_dim'] == 2
    assert resolved['nucleus_mask_dim'] == 3
    assert resolved['pathogen_mask_dim'] is None
    assert resolved['save_png'] is False
    assert resolved['png_size'] == [64, 64]


def test_the_manifest_it_writes_is_the_one_measure_reads_back(tmp_path, drawn):
    """``read_merged_plane_layout`` recomputes the dims and must agree."""
    from spacr.crops import read_merged_plane_layout

    found = assign_paths_by_regex(_paths(drawn), REGEX)
    destination = tmp_path / "project"
    written = write_field_table_project(found.table, str(destination))

    merged_dir = str(destination / "merged")
    layout = read_merged_plane_layout(merged_dir)
    assert layout['mask_dims'] == {'cell': 2, 'nucleus': 3}
    assert layout['intensity_channels'] == [0, 1]
    assert layout['mask_plane_order'] == ['cell', 'nucleus']

    with open(os.path.join(merged_dir, MERGED_LAYOUT_SIDECAR),
              encoding='utf-8') as handle:
        assert json.load(handle)['version'] == 1

    assert len(written['merged']) == 2
    stack = np.load(written['merged'][0])
    assert stack.shape == (48, 48, 4)
    assert stack.dtype == np.uint16
    assert set(np.unique(stack[..., 2])) == {0, 1}


def test_a_float_image_is_refused_rather_than_truncated(tmp_path):
    """Silently casting 0.5 to 0 would produce numbers that look measured."""
    folder = tmp_path / "floats"
    _write(str(folder / "fov001_C1.tif"),
           np.full((8, 8), 0.5, dtype=np.float32))
    cell = np.zeros((8, 8), np.uint16)
    cell[1:7, 1:7] = 1
    _write(str(folder / "fov001_cell_mask.tif"), cell)

    found = assign_paths_by_regex(_paths(folder), REGEX)
    assert found.table.problems() == []

    with pytest.raises(ConfigurationError, match="lose precision"):
        write_field_table_project(found.table, str(tmp_path / "out"))


def test_a_table_run_and_a_measure_run_agree(tmp_path, drawn):
    """The point of the whole instruction, checked both ways at once.

    Route A is the FEATURES button: a table of loose files, assigned by
    regex, run through :func:`measure_from_field_table`.

    Route B is the Measure module as it has always worked: a ``merged/``
    folder built by hand from the same arrays, run through
    :func:`measure_crop` with the mask dims spelled out.

    The two are compared on the folder tree they leave, the tables and
    columns in the database, and the measured values themselves.
    """
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    found = assign_paths_by_regex(_paths(drawn), REGEX)
    a_root = tmp_path / "by_table"
    by_table = measure_from_field_table(
        found.table, dict(LEAN), dst=str(a_root))

    b_root = tmp_path / "by_measure"
    merged_dir = b_root / "merged"
    os.makedirs(merged_dir, exist_ok=True)
    for index, field in enumerate((1, 2), start=1):
        c1, c2, cell, nucleus = _planes(field)
        np.save(str(merged_dir / f"drawn_A01_{index}.npy"),
                np.stack([c1, c2, cell, nucleus], axis=-1))
    settings = get_measure_crop_settings(dict(LEAN))
    settings.update({
        "src": str(merged_dir), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3, "pathogen_mask_dim": None,
        "crop_mode": ["cell"], "png_dims": [0, 1],
    })
    measure_crop(settings)

    a_db = by_table['db_path']
    b_db = str(b_root / "measurements" / "measurements.db")
    assert os.path.isfile(a_db)
    assert os.path.isfile(b_db)

    a_tables = _tables(a_db)
    b_tables = _tables(b_db)
    assert set(a_tables) == set(b_tables)
    for name in sorted(a_tables):
        assert a_tables[name][1] == b_tables[name][1], (
            f"{name} has different columns")
    for name in ("cell", "nucleus", "cytoplasm"):
        assert a_tables[name][0] == b_tables[name][0] > 0, (
            f"{name} has a different number of rows")

    a_tree = _tree(a_root)
    b_tree = _tree(b_root)
    assert {path for path in b_tree if path.startswith("data" + os.sep)} == \
        {path for path in a_tree if path.startswith("data" + os.sep)}
    assert {path for path in b_tree if path.startswith("merged" + os.sep)} <= \
        {path for path in a_tree if path.startswith("merged" + os.sep)}
    assert "measurements" + os.sep + "measurements.db" in a_tree

    with sqlite3.connect(a_db) as first, sqlite3.connect(b_db) as second:
        query = ('select prcf, object_label, cell_area, '
                 'cell_channel_0_mean_intensity from cell '
                 'order by prcf, object_label')
        assert first.execute(query).fetchall() == \
            second.execute(query).fetchall()


def test_one_drawn_mask_can_be_measured_against_several_acquisitions(tmp_path):
    """Scenario 3: the same mask file in several rows, different channels.

    Nothing special is needed for it -- a row owns a path, and two rows may
    own the same one -- but it is the scenario the instruction file says is
    hardest to reach from the Measure module, so it is pinned.
    """
    folder = tmp_path / "timecourse"
    cell = np.zeros((24, 24), np.uint16)
    cell[3:21, 3:21] = 1
    _write(str(folder / "shared_cell_mask.tif"), cell)
    for step in (1, 2, 3):
        _write(str(folder / f"t{step}_C1.tif"),
               np.full((24, 24), step * 100, np.uint16))

    table = FieldTable(rows=[], n_channels=1, roles=('cell',), plate='drawn')
    for step in (1, 2, 3):
        table.rows.append(FieldRow(
            label=f"t{step}",
            channels={0: str(folder / f"t{step}_C1.tif")},
            masks={'cell': str(folder / "shared_cell_mask.tif")},
            well='A01', field=step))

    assert table.problems() == []
    out = measure_from_field_table(
        table, dict(LEAN), dst=str(tmp_path / "project"))

    with sqlite3.connect(out['db_path']) as connection:
        rows = connection.execute(
            'select prcf, cell_channel_0_mean_intensity from cell '
            'order by prcf').fetchall()
    assert len(rows) == 3
    assert len({round(value, 3) for _stem, value in rows}) == 3


def test_a_second_drop_puts_the_same_channel_token_in_the_same_column():
    """The table remembers what column a token means; the drop does not decide.

    The FEATURES window's documented workflow is one field at a time -- draw,
    press FEATURES, move to the next image, draw again. Ranking each drop's
    tokens on their own made the ranking a property of the batch, so a second
    drop that happened not to carry ``C1`` put its ``C2`` in column 0 beside
    the first drop's ``C1``. Nothing on screen said so: the table read as
    complete, the run succeeded, and ``cell_channel_0_mean_intensity`` in the
    database was a different stain for different fields.
    """
    first = assign_paths_by_regex(
        ['/d/fov001_C1.tif', '/d/fov001_C2.tif', '/d/fov001_cell_mask.tif'],
        REGEX)
    assert first.table.channel_tokens == ('1', '2')
    assert first.table.rows[0].channels == {
        0: '/d/fov001_C1.tif', 1: '/d/fov001_C2.tif'}

    second = assign_paths_by_regex(
        ['/d/fov002_C2.tif', '/d/fov002_cell_mask.tif'], REGEX,
        table=first.table)

    rows = {row.label: row for row in second.table.rows}
    assert rows['fov002'].channels == {1: '/d/fov002_C2.tif'}, (
        "C2 is column 1 for every field or the database is not comparable")
    assert ('/d/fov002_C2.tif', 'fov002', 'channel 2') in second.assigned
    assert "fov002 has no file for channel 1." in second.table.problems()


def test_a_token_that_ranks_first_arriving_late_moves_the_columns_with_it():
    """A late ``C1`` renumbers the columns AND the files already in them.

    The alternative -- leaving the first drop's ``C2`` in column 0 and giving
    ``C1`` column 1 -- keeps every row agreeing, but it makes channel 0 the
    second stain for the whole table with nothing saying so. Re-ranking the
    union and moving what is already placed is what keeps "channel 0 is the
    lowest-ranked token present" true however the files arrived.
    """
    first = assign_paths_by_regex(
        ['/d/fov001_C2.tif', '/d/fov001_cell_mask.tif'], REGEX)
    assert first.table.rows[0].channels == {0: '/d/fov001_C2.tif'}

    second = assign_paths_by_regex(
        ['/d/fov001_C1.tif', '/d/fov002_C1.tif', '/d/fov002_C2.tif',
         '/d/fov002_cell_mask.tif'], REGEX, table=first.table)

    rows = {row.label: row for row in second.table.rows}
    assert second.table.channel_tokens == ('1', '2')
    assert rows['fov001'].channels == {
        0: '/d/fov001_C1.tif', 1: '/d/fov001_C2.tif'}
    assert rows['fov002'].channels == {
        0: '/d/fov002_C1.tif', 1: '/d/fov002_C2.tif'}
    assert second.table.n_channels == 2
    assert second.table.problems() == []


def test_renumbering_keeps_a_file_that_was_browsed_into_a_column():
    """A cell filled by browsing is not thrown away when the columns move.

    No token claims that column, so it cannot be ranked; it is given a column
    after the tokened ones instead of being dropped, because a file the user
    put somewhere by hand disappearing is the one failure this table cannot
    afford.
    """
    first = assign_paths_by_regex(
        ['/d/fov001_C2.tif', '/d/fov001_cell_mask.tif'], REGEX)
    first.table.n_channels = 2
    first.table.rows[0].channels[1] = '/d/hand_picked.tif'

    second = assign_paths_by_regex(['/d/fov001_C1.tif'], REGEX,
                                   table=first.table)

    channels = second.table.rows[0].channels
    assert channels[0] == '/d/fov001_C1.tif'
    assert channels[1] == '/d/fov001_C2.tif'
    assert '/d/hand_picked.tif' in channels.values()
    assert sorted(channels) == [0, 1, 2]


def test_the_destination_a_run_uses_is_the_one_the_window_can_show(tmp_path):
    """``field_table_destination`` is the single answer, derived or given.

    ``src`` is shown in the window as a disabled box captioned "the table
    decides this", so the window and the run having separate opinions about
    where the output goes means the window names a folder the results are not
    in.
    """
    from spacr.measure import field_table_destination

    folder = tmp_path / "drawn"
    _write(str(folder / "fov001_C1.tif"), np.ones((8, 8), np.uint16))
    table = FieldTable(
        rows=[FieldRow(label='fov001',
                       channels={0: str(folder / "fov001_C1.tif")},
                       masks={'cell': 'x'}, well='A01', field=1)],
        n_channels=1, roles=('cell',), plate='drawn')

    assert field_table_destination(table, None) == str(folder / "features")
    assert field_table_destination(table, tmp_path) == str(tmp_path)
    assert field_table_destination(
        FieldTable(rows=[], n_channels=1, roles=('cell',)), None) is None

    settings = field_table_settings(
        table, {}, dst=field_table_destination(table, None))
    assert settings['src'] == os.path.join(
        str(folder / "features"), 'merged')


def test_progress_reports_both_stages_of_a_run(tmp_path):
    """Writing the arrays and measuring them are announced separately.

    On a table of many fields the measuring takes minutes and the writing
    does not, so one "working" line covering both tells the user nothing
    about which of them they are waiting for.
    """
    folder = tmp_path / "drawn"
    yy, xx = np.indices((24, 24))
    _write(str(folder / "fov001_C1.tif"), ((yy + xx) % 4096).astype(np.uint16))
    cell = np.zeros((24, 24), np.uint16)
    cell[3:21, 3:21] = 1
    _write(str(folder / "fov001_cell_mask.tif"), cell)

    table = FieldTable(
        rows=[FieldRow(label='fov001',
                       channels={0: str(folder / "fov001_C1.tif")},
                       masks={'cell': str(folder / "fov001_cell_mask.tif")},
                       well='A01', field=1)],
        n_channels=1, roles=('cell',), plate='drawn')

    said = []
    measure_from_field_table(table, dict(LEAN), dst=str(tmp_path / "out"),
                             progress=said.append)

    assert len(said) == 2
    assert "Writing" in said[0]
    assert "Measuring" in said[1]


def test_a_progress_callback_that_raises_does_not_fail_the_run(tmp_path):
    """The window may be closed mid-run; the run is not its listener's problem.

    The callback the FEATURES window passes emits a Qt signal, and a worker
    parked past its widget's destruction raises ``RuntimeError`` from the
    emit. A measurement that was going to succeed must not be lost because
    nobody is watching it any more.
    """
    folder = tmp_path / "drawn"
    _write(str(folder / "fov001_C1.tif"), np.ones((24, 24), np.uint16) * 7)
    cell = np.zeros((24, 24), np.uint16)
    cell[3:21, 3:21] = 1
    _write(str(folder / "fov001_cell_mask.tif"), cell)

    table = FieldTable(
        rows=[FieldRow(label='fov001',
                       channels={0: str(folder / "fov001_C1.tif")},
                       masks={'cell': str(folder / "fov001_cell_mask.tif")},
                       well='A01', field=1)],
        n_channels=1, roles=('cell',), plate='drawn')

    def angry(_message):
        raise RuntimeError("Signal source has been deleted")

    out = measure_from_field_table(table, dict(LEAN),
                                   dst=str(tmp_path / "out"), progress=angry)
    assert os.path.isfile(out['db_path'])
