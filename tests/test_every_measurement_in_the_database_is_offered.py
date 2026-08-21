"""187 A: compare any measurement from the database tables.

    "compare any measurement from the database tables (requires them to be
    joined)"

THE JOIN IS THE PRECONDITION AND IT IS SAID OUT LOUD. The montage's object
rows come out of `png_list`, which holds the crop path and the classification
score; every morphological measurement -- cell, nucleus, pathogen, cytoplasm
-- is in the object tables beside it. Offering the short list without saying
why it is short is what this instruction is against.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (join_measurements,
                                            measurements_are_joined,
                                            object_identity)

CELLS = 12


def _database(path) -> str:
    """A measurements.db with png_list and a cell table, joinable."""
    frame = pd.DataFrame({
        "plateID": ["plate1"] * CELLS,
        "rowID": ["A"] * CELLS,
        "columnID": ["01"] * CELLS,
        "fieldID": ["1"] * CELLS,
        "prcf": ["plate1_A_01_1"] * CELLS,
    })
    # `cell_id` on png_list and `object_label` on the cell table is what the
    # Measure module writes and what `_read_and_join_tables` joins on.
    png = frame.assign(
        cell_id=[str(i + 1) for i in range(CELLS)],
        png_path=[f"/nowhere/{i}.png" for i in range(CELLS)],
        pred=np.linspace(0.1, 0.9, CELLS))
    cell = frame.assign(
        object_label=[i + 1 for i in range(CELLS)],
        cell_area=np.arange(CELLS, dtype=float) * 10.0,
        cell_perimeter=np.arange(CELLS, dtype=float) + 3.0,
        cell_channel_1_mean_intensity=np.linspace(5.0, 9.0, CELLS))
    connection = sqlite3.connect(str(path))
    try:
        png.to_sql("png_list", connection, index=False)
        cell.to_sql("cell", connection, index=False)
    finally:
        connection.close()
    return str(path)


@pytest.fixture
def database(tmp_path):
    return _database(tmp_path / "measurements.db")


@pytest.fixture
def objects():
    """What the montage holds: png_list's columns and nothing else."""
    return pd.DataFrame({
        "plateID": ["plate1"] * CELLS,
        "rowID": ["A"] * CELLS,
        "columnID": ["01"] * CELLS,
        "fieldID": ["1"] * CELLS,
        "object_label": [str(i + 1) for i in range(CELLS)],
        "prcf": ["plate1_A_01_1"] * CELLS,
        "prc": ["plate1_A_01"] * CELLS,
        "pred": np.linspace(0.1, 0.9, CELLS),
    })


class TestAnObjectIsIdentifiedTheSameWayOnBothSides:

    def test_prcfo_is_used_when_it_is_there(self):
        frame = pd.DataFrame({"prcfo": ["a_b_c_1_5"], "label": ["9"]})

        assert list(object_identity(frame)) == ["a_b_c_1_5"]

    def test_it_is_composed_from_prcf_and_a_label(self, objects):
        assert list(object_identity(objects))[0] == "plate1_A_01_1_1"

    def test_prc_and_a_field_will_do(self):
        frame = pd.DataFrame({"prc": ["plate1_A_01"], "fieldID": ["1"],
                              "object_label": ["7"]})

        assert list(object_identity(frame)) == ["plate1_A_01_1_7"]

    def test_every_label_spelling_is_read(self):
        for column in ("object_label", "label", "cell_id", "nucleus_id",
                       "pathogen_id", "cytoplasm_id"):
            frame = pd.DataFrame({"prcf": ["p_A_01_1"], column: ["3"]})
            assert list(object_identity(frame)) == ["p_A_01_1_3"], column

    def test_no_label_is_none(self):
        frame = pd.DataFrame({"prcf": ["p_A_01_1"], "pred": [0.5]})

        assert object_identity(frame) is None

    def test_no_well_is_none(self):
        assert object_identity(pd.DataFrame({"object_label": ["3"]})) is None


class TestTheJoinBringsTheMeasurementsAcross:

    def test_the_columns_arrive(self, objects, database):
        wide, note = join_measurements(objects, [database])

        assert "cell_area" in wide.columns
        assert "cell_perimeter" in wide.columns
        assert note == ""

    def test_the_values_land_on_the_right_cells(self, objects, database):
        wide, _note = join_measurements(objects, [database])

        assert list(wide["cell_area"]) == [i * 10.0 for i in range(CELLS)]

    def test_the_index_is_untouched(self, objects, database):
        """The caller's groups are index values into THIS frame; a reindexed
        copy would re-annotate different cells."""
        objects.index = pd.Index([f"cell-{i}" for i in range(CELLS)])

        wide, _note = join_measurements(objects, [database])

        assert list(wide.index) == list(objects.index)

    def test_a_repeated_index_does_not_multiply_the_rows(self, objects,
                                                         database):
        """`_all_objects` concatenates one frame per plan, so the index
        repeats -- and a join on a repeated label is a cartesian product."""
        doubled = pd.concat([objects, objects])

        wide, _note = join_measurements(doubled, [database])

        assert len(wide) == 2 * CELLS

    def test_the_columns_the_montage_selected_on_are_not_replaced(
            self, objects, database):
        before = list(objects["pred"])

        wide, _note = join_measurements(objects, [database])

        assert list(wide["pred"]) == before

    def test_an_unmatched_row_is_counted_out_loud(self, objects, database):
        objects.loc[0, "object_label"] = "999"

        wide, note = join_measurements(objects, [database])

        assert "1 of 12 object row(s) found no match" in note
        assert pd.isna(wide["cell_area"].iloc[0])

    def test_an_unreadable_database_is_a_sentence_not_a_traceback(
            self, objects, tmp_path):
        wide, note = join_measurements(objects, [str(tmp_path / "gone.db")])

        assert "no measurement table could be read" in note
        assert list(wide.columns) == list(objects.columns)

    def test_rows_with_no_identity_say_so(self, database):
        frame = pd.DataFrame({"pred": [0.1, 0.2]})

        wide, note = join_measurements(frame, [database])

        assert "no object identity" in note
        assert list(wide.columns) == ["pred"]

    def test_two_databases_are_both_read(self, objects, tmp_path):
        first = _database(tmp_path / "one.db")
        second = _database(tmp_path / "two.db")

        wide, note = join_measurements(objects, [first, second])

        assert "cell_area" in wide.columns
        assert len(wide) == CELLS


class TestThePanelCanTellWhetherItNeedsTheJoin:

    def test_png_list_rows_are_not_joined(self, objects):
        assert not measurements_are_joined(objects)

    def test_joined_rows_are(self, objects, database):
        wide, _note = join_measurements(objects, [database])

        assert measurements_are_joined(wide)

    def test_an_id_column_is_not_a_measurement(self):
        """`cell_id` is how a nucleus names its cell, not a measurement."""
        frame = pd.DataFrame({"cell_id": [1], "nucleus_id": [2]})

        assert not measurements_are_joined(frame)
