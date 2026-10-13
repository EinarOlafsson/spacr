"""A merged database keeps the gRNA annotation (instruction 203).

"when i merge the databases and pick a database measurement to compare i am
not able to compare the gRNA annotated groups any longer."

SETTLE IT BY COUNTING, not by reading the code: how many objects carry an
annotation before the merge, and how many carry one after. A merge that
turns 1,536 into 0 is a join key; one that turns 1,536 into 1,536 and still
shows no groups is the compare screen.

The answer was BOTH. The numeric filter that keeps a merge from dragging
every path across was also dropping the gRNA column -- a string, and the
grouping the whole screen exists to serve -- and a join that matched nothing
was handed on as a frame of all-NaN measurements for a panel to draw empty.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (ANNOTATION_COLUMNS,
                                            join_measurements)


def _database(path, prcfo, *, extra=None):
    """A measurements.db with a cell table keyed the way spaCR writes one."""
    frame = pd.DataFrame({
        "prcfo": list(prcfo),
        "cell_area": np.linspace(400, 600, len(prcfo)),
        "cell_channel_1_mean_intensity": np.linspace(900, 1100, len(prcfo)),
    })
    if extra:
        for name, values in extra.items():
            frame[name] = values
    with sqlite3.connect(str(path)) as handle:
        frame.to_sql("cell", handle, index=False)
    return str(path)


@pytest.fixture
def objects():
    """Prediction rows: an object identity and a gRNA annotation."""
    prcfo = [f"plate1_r1_c{c}_f1_o{o}" for c in (1, 2) for o in range(1, 9)]
    return pd.DataFrame({
        "prcfo": prcfo,
        "grna": (["g1"] * 8) + (["g2"] * 8),
        "pred": np.linspace(0.1, 0.9, len(prcfo)),
    })


class TestTheCountSurvives:
    """"The count of annotated objects after the merge equals the count
    before it"."""

    def test_it_does(self, objects, tmp_path):
        db = _database(tmp_path / "measurements.db", objects["prcfo"])
        before = int(objects["grna"].notna().sum())
        merged, note = join_measurements(objects, [db])
        after = int(merged["grna"].notna().sum())
        assert after == before, note

    def test_and_the_measurements_arrived(self, objects, tmp_path):
        db = _database(tmp_path / "measurements.db", objects["prcfo"])
        merged, note = join_measurements(objects, [db])
        assert "cell_area" in merged.columns, note
        assert merged["cell_area"].notna().all()

    def test_the_groups_are_still_there_to_compare(self, objects, tmp_path):
        db = _database(tmp_path / "measurements.db", objects["prcfo"])
        merged, _ = join_measurements(objects, [db])
        assert set(merged["grna"]) == {"g1", "g2"}


class TestAnAnnotationInTheDatabaseComesAcross:
    """The numeric filter was dropping it: a gRNA name is a string."""

    def test_a_string_annotation_column_is_kept(self, objects, tmp_path):
        db = _database(tmp_path / "measurements.db", objects["prcfo"],
                       extra={"condition": ["nc", "pc"] * 8})
        merged, note = join_measurements(objects, [db])
        assert "condition" in merged.columns, note
        assert set(merged["condition"].dropna()) == {"nc", "pc"}

    def test_an_ordinary_string_column_is_still_left_out(self, objects,
                                                         tmp_path):
        """A merge that brought paths and filenames over would put hundreds
        of useless entries in the measurement chooser."""
        db = _database(tmp_path / "measurements.db", objects["prcfo"],
                       extra={"png_path": ["/x/a.png"] * 16})
        merged, _ = join_measurements(objects, [db])
        assert "png_path" not in merged.columns

    def test_the_list_is_named_rather_than_every_object_column(self):
        assert "grna" in ANNOTATION_COLUMNS
        assert "png_path" not in ANNOTATION_COLUMNS


class TestASilentZeroIsTheRealFault:
    """"A join that matches nothing raises or is reported, never drawn as
    empty"."""

    def test_no_overlap_is_reported(self, objects, tmp_path):
        elsewhere = [f"plate9_r9_c9_f9_o{o}" for o in range(1, 17)]
        db = _database(tmp_path / "measurements.db", elsewhere)
        _, note = join_measurements(objects, [db])
        assert "MATCHED NOTHING" in note

    def test_the_note_says_it_is_a_join_key(self, objects, tmp_path):
        """"there is no run where zero of the objects have a prediction"."""
        elsewhere = [f"plate9_r9_c9_f9_o{o}" for o in range(1, 17)]
        db = _database(tmp_path / "measurements.db", elsewhere)
        _, note = join_measurements(objects, [db])
        assert "join-key" in note

    def test_the_original_rows_come_back_untouched(self, objects, tmp_path):
        """Handing on a frame of all-NaN measurement columns is how the empty
        plot gets drawn."""
        elsewhere = [f"plate9_r9_c9_f9_o{o}" for o in range(1, 17)]
        db = _database(tmp_path / "measurements.db", elsewhere)
        merged, _ = join_measurements(objects, [db])
        assert list(merged.columns) == list(objects.columns)
        assert len(merged) == len(objects)

    def test_a_partial_match_is_reported_but_kept(self, objects, tmp_path):
        """Some rows unmatched is a real screen; none is a bug."""
        half = list(objects["prcfo"])[:8]
        db = _database(tmp_path / "measurements.db", half)
        merged, note = join_measurements(objects, [db])
        assert "cell_area" in merged.columns
        assert "found no match" in note
        assert int(merged["cell_area"].notna().sum()) == 8


class TestItIsNotAFixtureBuiltToMatch:
    """The identities go through a real sqlite round trip and spaCR's own
    identity resolution, not a hand-built key on both sides."""

    def test_the_database_is_read_back_by_spacr(self, objects, tmp_path):
        db = _database(tmp_path / "measurements.db", objects["prcfo"])
        merged, note = join_measurements(objects, [db])
        assert "cell_area" in merged.columns, note

    def test_two_databases_merge_into_one_frame(self, objects, tmp_path):
        first = list(objects["prcfo"])[:8]
        second = list(objects["prcfo"])[8:]
        one = _database(tmp_path / "a.db", first)
        two = _database(tmp_path / "b.db", second)
        merged, note = join_measurements(objects, [one, two])
        assert int(merged["cell_area"].notna().sum()) == 16, note
