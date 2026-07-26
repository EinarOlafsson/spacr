"""Plates past 384 — every row, however far past P, reaches the figure.

MEASURED before the fix, on a 1536-well plate (32 rows ``A``…``AF`` by 48
columns) whose crop names were written by the real writer
:func:`spacr.utils.filepaths_to_database`::

    generate_plate_heatmap(frame, 'plate1', 'value', 'count', 'all', 0)[0].shape
    (16, 27)                       # 432 of the plate's 1536 wells

1104 measured wells — every one of them in the database, every one of them
parsed correctly by :mod:`spacr.schema` on the way in — were absent from the
picture, and nothing was printed, logged or raised.
:func:`spacr.plot.generate_plate_heatmap` built its axes from two literals::

    row_order = [f'r{i}' for i in range(1, 17)]     # A..P
    col_order = [f'c{i}' for i in range(1, 28)]

and made them the categories of a ``pd.Categorical``. A label outside the
categories becomes NaN, and a NaN group key is dropped by ``groupby``: a
silent delete keyed on a plate geometry that was decided by the size of the
1996 catalogue rather than by the data in front of it.

Nothing in the suite pinned those two lines, which is a large part of why
they survived; this file pins their replacement. The rule now is that the
axes are **read off the data** — so a 96 plate is still 8x12 and a 384 is
still 16x24, nothing is padded out to the largest format that exists — and a
row that genuinely cannot be placed is **named in a report** rather than
quietly deleted. Swapping one silent drop for a quieter one would fix
nothing.

Everything here is CPU-only, offline, and built by the real writers rather
than by hand: the well identities under test are the ones spaCR actually
stores for ``AF48``, not the ones a test frame asserts it ought to.
"""
from __future__ import annotations

import logging
import sqlite3

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from spacr import plate_qc, schema
from spacr.errors import ConfigurationError
from spacr.plot import generate_plate_heatmap, plot_plates
from spacr.utils import _merge_and_save_to_database, filepaths_to_database


@pytest.fixture(autouse=True)
def _no_figure_leak():
    plt.close("all")
    yield
    plt.close("all")


def _plate_frame(tmp_path, n_rows, n_cols, plate="plate1", objects=1):
    """A long per-object frame for an ``n_rows`` x ``n_cols`` plate.

    The crop names go through :func:`spacr.utils.filepaths_to_database` —
    the writer that puts ``png_list`` into a real ``measurements.db`` — and
    the well keys come back out of SQLite. So ``rowID`` for well ``AF48`` is
    whatever spaCR stores for it, and a test that passes here is a statement
    about the pipeline rather than about a hand-built frame.

    Each well carries one distinct value, ``row * 100 + column``, so a well
    drawn in the wrong place is visible as a wrong number rather than as a
    plausible one.
    """
    root = tmp_path / f"exp_{n_rows}x{n_cols}"
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    paths = [str(root / "cell_png"
                 / f"{plate}_{schema.well_id(r, c)}_1_{obj}.png")
             for r in range(1, n_rows + 1)
             for c in range(1, n_cols + 1)
             for obj in range(1, objects + 1)]
    filepaths_to_database(paths, {"timelapse": False}, str(root), "cell")

    conn = sqlite3.connect(root / "measurements" / "measurements.db")
    try:
        png = pd.read_sql_query("SELECT * FROM png_list", conn)
    finally:
        conn.close()

    png["prc"] = [schema.compose_prc(p, r, c) for p, r, c
                  in zip(png["plateID"], png["rowID"], png["columnID"])]
    png["value"] = [float(schema.row_index(r) * 100 + schema.column_index(c))
                    for r, c in zip(png["rowID"], png["columnID"])]
    return png


# ---------------------------------------------------------------------------
# The whole plate, whatever size it is
# ---------------------------------------------------------------------------

def test_a_1536_plate_reaches_the_figure_whole(tmp_path):
    """32 rows and 48 columns, and AF48 among them. Was (16, 27) — 432 wells."""
    frame = _plate_frame(tmp_path, 32, 48)
    assert len(frame) == 1536                       # the writer kept them all

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    assert plate_map.shape == (32, 48)
    assert plate_map.size == 1536
    assert plate_map.index.tolist() == [f"r{i}" for i in range(1, 33)]
    assert plate_map.columns.tolist() == [f"c{i}" for i in range(1, 49)]

    # The far corner, named as a plate names it.
    assert schema.well_id("r32", "c48") == "AF48"
    assert plate_map.loc["r32", "c48"] == 32 * 100 + 48
    # ...and every well holds its own measurement, so nothing was shuffled
    # and no cell was invented by fillna to pad the grid out.
    expected = np.array([[r * 100 + c for c in range(1, 49)]
                         for r in range(1, 33)], dtype=float)
    assert np.array_equal(plate_map.to_numpy(), expected)
    assert (vmin, vmax) == (101.0, 3248.0)


@pytest.mark.parametrize("n_rows,n_cols,n_wells", [(8, 12, 96), (16, 24, 384)])
def test_a_smaller_plate_keeps_its_own_size(tmp_path, n_rows, n_cols, n_wells):
    """Reading the axes off the data must not mean padding to the biggest format."""
    frame = _plate_frame(tmp_path, n_rows, n_cols)
    plate_map, _ = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    assert plate_map.shape == (n_rows, n_cols)
    assert plate_map.size == n_wells
    assert plate_map.index[-1] == f"r{n_rows}"
    assert plate_map.columns[-1] == f"c{n_cols}"
    assert plate_map.loc[f"r{n_rows}", f"c{n_cols}"] == n_rows * 100 + n_cols


def test_the_first_well_past_the_old_grid_is_drawn(tmp_path):
    """``Q28`` — row 17, column 28 — is the smallest well the literals deleted."""
    frame = _plate_frame(tmp_path, 17, 28)
    plate_map, _ = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    assert schema.well_id("r17", "c28") == "Q28"
    assert "r17" in plate_map.index and "c28" in plate_map.columns
    assert plate_map.loc["r17", "c28"] == 17 * 100 + 28
    assert plate_map.shape == (17, 28)


def test_an_arbitrary_row_count_past_af_is_still_a_grid(tmp_path):
    """Nothing stops at 1536 either — 40 rows means rows up to ``AN``."""
    frame = _plate_frame(tmp_path, 40, 50)
    plate_map, _ = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    assert plate_map.shape == (40, 50)
    assert schema.well_id("r40", "c50") == "AN50"
    assert plate_map.loc["r40", "c50"] == 40 * 100 + 50


# ---------------------------------------------------------------------------
# Rows past Z
# ---------------------------------------------------------------------------

def test_rows_past_z_round_trip_from_the_writer_to_the_figure(tmp_path):
    """``AA``/``AB``/``AF`` are ordinary 1536-plate rows, not an edge case."""
    frame = _plate_frame(tmp_path, 32, 48)
    plate_map, _ = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    for letters, index in (("AA", 27), ("AB", 28), ("AF", 32)):
        # the letter walk agrees with itself...
        assert schema.row_index_from_letters(letters) == index
        assert schema.letters_from_row_index(index) == letters
        assert plate_qc.parse_row_label(letters) == index
        # ...the writer stored a whole row of them...
        assert int((frame["rowID"] == f"r{index}").sum()) == 48
        # ...and the plotter drew it.
        assert plate_map.loc[f"r{index}", "c1"] == index * 100 + 1
        assert schema.well_id(f"r{index}", 1) == f"{letters}01"


def test_the_row_letter_walk_never_leaves_the_alphabet():
    """``chr(ord('A') + n)`` gives ``'['`` for row 27. That is the whole bug."""
    assert chr(ord("A") + 27 - 1) == "["          # what the naive walk emitted
    assert schema.letters_from_row_index(27) == "AA"

    for index in range(1, 1001):
        letters = schema.letters_from_row_index(index)
        assert letters.isalpha() and letters.isupper(), (index, letters)
        assert schema.row_index_from_letters(letters) == index
        # plate_qc labels the axes of the Qt plate; it must not disagree.
        assert plate_qc.row_label(index) == letters


def test_every_row_the_1536_heatmap_draws_has_a_well_name(tmp_path):
    frame = _plate_frame(tmp_path, 32, 48)
    plate_map, _ = generate_plate_heatmap(
        frame, "plate1", "value", "mean", "all", 0)

    wells = [schema.well_id(r, c)
             for r in plate_map.index for c in plate_map.columns]
    assert len(set(wells)) == 1536
    assert all(w[:-2].isalpha() and w[-2:].isdigit() for w in wells)
    assert wells[0] == "A01" and wells[-1] == "AF48"


# ---------------------------------------------------------------------------
# A well that cannot be placed is reported
# ---------------------------------------------------------------------------

def _frame_with_a_broken_identifier(tmp_path):
    """A real 2x3 plate plus the two identifiers spaCR itself can produce.

    ``'error'`` is what ``utils._map_wells`` writes into every slot when a
    name will not parse, and those strings go into the database as if they
    were an identity; a two-token ``prc`` is what a truncated identifier
    looks like. Neither names a position on any plate.
    """
    frame = _plate_frame(tmp_path, 2, 3)[["prc", "value"]]
    broken = pd.DataFrame({"prc": ["plate1_error_error", "plate1_r4"],
                           "value": [1.0, 2.0]})
    return pd.concat([frame, broken], ignore_index=True)


def test_a_well_that_cannot_be_placed_is_named_not_dropped(tmp_path, caplog):
    frame = _frame_with_a_broken_identifier(tmp_path)

    with caplog.at_level(logging.ERROR, logger="spacr.errors"):
        plate_map, _ = generate_plate_heatmap(
            frame, "plate1", "value", "mean", "all", 0)

    # The real wells are all still drawn — reporting is not refusing.
    assert plate_map.shape == (2, 3)
    assert plate_map.loc["r2", "c3"] == 2 * 100 + 3

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "plate1_error_error" in message
    assert "plate1_r4" in message
    assert "missing from the heatmap" in message
    assert "2 row(s)" in message


def test_strict_errors_turns_an_unplaceable_well_into_a_stop(tmp_path, monkeypatch):
    """``SPACR_STRICT_ERRORS=1`` is for the runs that would rather not guess."""
    monkeypatch.setenv("SPACR_STRICT_ERRORS", "1")
    frame = _frame_with_a_broken_identifier(tmp_path)

    with pytest.raises(ConfigurationError, match="plate1_error_error"):
        generate_plate_heatmap(frame, "plate1", "value", "mean", "all", 0)


def test_a_plate_of_nothing_but_unplaceable_wells_is_empty_and_says_so(caplog):
    frame = pd.DataFrame({"prc": ["plate1_error_error"] * 3,
                          "value": [1.0, 2.0, 3.0]})
    with caplog.at_level(logging.ERROR, logger="spacr.errors"):
        plate_map, limits = generate_plate_heatmap(
            frame, "plate1", "value", "mean", "all", 0)

    assert plate_map.values.size == 0
    assert limits == (0.0, 1.0)
    assert "plate1_error_error" in "\n".join(
        r.getMessage() for r in caplog.records)


def test_an_empty_frame_is_an_empty_plate_not_an_exception():
    """The old body opened with ``df['prc'].iloc[0]`` — IndexError on no rows."""
    empty = pd.DataFrame({"prc": pd.Series([], dtype=object),
                          "value": pd.Series([], dtype=float)})
    plate_map, limits = generate_plate_heatmap(
        empty, "plate1", "value", "mean", "all", 0)
    assert plate_map.values.size == 0
    assert limits == (0.0, 1.0)


def test_a_readable_plate_reports_nothing(tmp_path, caplog):
    """No false alarms: a clean 1536 plate must be silent."""
    frame = _plate_frame(tmp_path, 32, 48)
    with caplog.at_level(logging.ERROR, logger="spacr.errors"):
        generate_plate_heatmap(frame, "plate1", "value", "mean", "all", 0)
    assert caplog.records == []


# ---------------------------------------------------------------------------
# prc is read right to left
# ---------------------------------------------------------------------------

def test_a_frame_mixing_prc_lengths_places_every_row():
    """Only ``prc.iloc[0]`` was probed, so the minority shape misaligned.

    MEASURED with the old body on these two rows::

        df['plateID'], df['rowID'], df['columnID'] = zip(*df['prc'].str.split('_'))
        [{'plateID': 'plate1', 'rowID': 'r1',     'columnID': 'c1'},
         {'plateID': 'exp1',   'rowID': 'plate1', 'columnID': 'r2'}]

    ``zip`` stops at the shortest identifier, so the 4-token row had its
    plate read as a row, its row read as a column, and ``c3`` dropped
    entirely — after which the Categorical deleted it for good measure.
    """
    df = pd.DataFrame({"prc": ["plate1_r1_c1", "exp1_plate1_r2_c3"],
                       "value": [1.0, 2.0]})
    plate_map, _ = generate_plate_heatmap(
        df, "plate1", "value", "mean", "all", 0)

    assert plate_map.loc["r1", "c1"] == 1.0
    assert plate_map.loc["r2", "c3"] == 2.0


def test_letter_rows_in_a_prc_are_the_same_rows_as_prefixed_ones():
    """``plate1_AA_1`` and ``plate1_r27_c1`` are one well, not two."""
    df = pd.DataFrame({"prc": ["plate1_AA_1", "plate1_r27_c1"],
                       "value": [4.0, 6.0]})
    plate_map, _ = generate_plate_heatmap(
        df, "plate1", "value", "mean", "all", 0)

    assert plate_map.shape == (1, 1)
    assert plate_map.loc["r27", "c1"] == 5.0          # mean(4, 6)


# ---------------------------------------------------------------------------
# The figure itself
# ---------------------------------------------------------------------------

def test_plot_plates_draws_all_1536_cells(tmp_path):
    """Seaborn thins the tick labels on a big plate; the mesh is the data."""
    frame = _plate_frame(tmp_path, 32, 48)
    fig = plot_plates(frame, "value", "mean", "all", "viridis",
                      min_count=0, verbose=False)

    heat = next(a for a in fig.axes if a.get_title() == "plate1")
    mesh = heat.collections[0].get_array()
    assert mesh.shape == (32, 48)
    assert mesh.size == 1536
    # bottom-right cell of the mesh is well AF48
    assert float(mesh[31, 47]) == 32 * 100 + 48
    assert float(mesh[0, 0]) == 1 * 100 + 1


# ---------------------------------------------------------------------------
# The rest of spaCR agrees about the same plate
# ---------------------------------------------------------------------------

def test_plate_qc_lays_out_the_same_1536_plate(tmp_path):
    """The QC grid and the plot grid must not disagree about where AF48 is."""
    frame = _plate_frame(tmp_path, 32, 48)
    layout = plate_qc.plate_layout(frame, "value", grouping="mean")

    assert layout.attrs["plate_format"] == 1536
    assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == (32, 48)
    assert len(layout) == 1536
    assert {"A01", "AA01", "AF48"} <= set(layout["well"])

    grid = plate_qc.layout_matrix(layout)
    assert grid.shape == (32, 48)
    assert grid.index.tolist()[-1] == "AF"
    assert grid.loc["AF", 48] == 32 * 100 + 48


def test_the_object_writer_stores_a_far_corner_well_as_a_far_corner_well(tmp_path):
    """``_merge_and_save_to_database`` keys ``AF48`` on row 32, not on an error."""
    root = tmp_path / "objects"
    (root / "measurements").mkdir(parents=True)
    _merge_and_save_to_database(
        pd.DataFrame({"label": [1], "cell_area": [10.0]}),
        pd.DataFrame({"label": [1], "cell_channel_0_mean_intensity": [2.0]}),
        "cell", str(root), "plate1_AF48_1", "exp")

    conn = sqlite3.connect(root / "measurements" / "measurements.db")
    try:
        cell = pd.read_sql_query("SELECT * FROM cell", conn)
    finally:
        conn.close()

    row = cell.iloc[0]
    assert (row["rowID"], row["columnID"]) == ("r32", "c48")
    assert row["prcf"] == "plate1_r32_c48_f1"
    assert schema.well_id(row["rowID"], row["columnID"]) == "AF48"


# ---------------------------------------------------------------------------
# Folder metadata
# ---------------------------------------------------------------------------

def test_a_well_folder_past_p_is_recognised_as_a_well():
    """``_WELL_RX`` was ``[A-P]\\d{1,3}``: the bottom half of a 1536 plate."""
    from spacr.qt.folder_metadata import _classify

    for token in ("A01", "P24", "Q01", "Z12", "AA01", "AF48", "a1"):
        assert _classify(token) == "well", token
    # The explicit recognisers are still checked first and still win.
    assert _classify("F01") == "field"
    assert _classify("field_2") == "field"
    assert _classify("C01") == "channel"
    assert _classify("ch2") == "channel"
    assert _classify("plate3") == "plate"
    assert _classify("not_a_well") is None


def test_synthetic_well_names_keep_counting_past_p():
    """385 folders means row Q, not a second ``E001``."""
    from spacr.qt.folder_metadata import _well_from_index

    names = [_well_from_index(i) for i in range(1536)]
    assert len(set(names)) == 1536
    assert names[0] == "A01" and names[383] == "P24" and names[384] == "Q01"
    assert all(schema.parse_well(n) != (n, n) for n in names)
