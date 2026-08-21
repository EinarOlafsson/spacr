"""196 C: the console's first line reports the design it actually has.

Reported 2026-08-21, in the log of the run that prompted 196:

    The count files hold 1380 distinct gRNAs in 642551 rows. no 'prc',
    'plate_row' or 'rowID'/'columnID' column, so wells were not counted

The count tables of the example screen name 1,536 wells perfectly well. Two
faults stacked, and both are the kind that print a confident number rather
than fail:

  1. THE SCAN READ RAW. `pd.read_csv`, so a file spelling its keys
     `row_name` / `column_name` -- which is what these do, and why 145
     converted the readers around it -- carried no column the scan
     recognised.
  2. AND THE FALLBACK GUESSED A PLATE. With no `plateID` column it
     substituted the literal "plate1", so all four files' wells collapsed
     onto the same 384 names. `_well_keys`' own docstring forbids exactly
     that: "a well count off by the number of plates is worse than no well
     count."
"""
from __future__ import annotations

import glob

import pandas as pd
import pytest

from spacr.qt.screens.settings_model import _well_keys, regression_design_scan


def _counts(path, plate=None, rows=("r1", "r2"), columns=("c1", "c2")):
    """A count table in the spelling the example screen uses."""
    out = []
    for row in rows:
        for column in columns:
            for guide in range(3):
                entry = {"row_name": row, "column_name": column,
                         "grna_name": f"TGGT1_000{guide}00_{guide + 1}",
                         "count": 100 + guide}
                if plate is not None:
                    entry["plate"] = plate
                out.append(entry)
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


class TestItReadsTheLegacySpelling:

    def test_row_name_and_column_name_are_wells(self, tmp_path):
        """The scan reads through `spacr.tabular` now, so the canonical
        names arrive whatever the file spells them."""
        got = regression_design_scan(
            {"count_data": [_counts(tmp_path / "a.csv")]})

        assert got["wells"] == 4
        assert "were not counted" not in got["note"]

    def test_it_still_counts_the_guides_and_genes(self, tmp_path):
        got = regression_design_scan(
            {"count_data": [_counts(tmp_path / "a.csv")]})

        assert got["guides"] == 3
        assert got["genes"] == 3


class TestOneFileIsOnePlate:
    """`load_regression_input_pairs`' rule for the same question: plate
    identity comes from the pair-row order when neither side declares it."""

    def test_two_files_of_the_same_wells_are_two_plates_of_wells(self,
                                                                 tmp_path):
        files = [_counts(tmp_path / "a.csv"), _counts(tmp_path / "b.csv")]

        got = regression_design_scan({"count_data": files})

        assert got["wells"] == 8, "the two files' wells collapsed into one"
        assert got["files"] == 2

    def test_a_declared_plate_still_wins(self, tmp_path):
        """A file that says which plate it is decides for itself."""
        files = [_counts(tmp_path / "a.csv", plate="plateX"),
                 _counts(tmp_path / "b.csv", plate="plateX")]

        got = regression_design_scan({"count_data": files})

        assert got["wells"] == 4, "the declared plate was overridden"

    def test_the_key_helper_does_not_invent_a_plate(self):
        """`_well_keys`' own docstring: a well count off by the number of
        plates is worse than no well count."""
        frame = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"]})
        frame.attrs["spacr_scan_plate"] = "plate3"

        assert list(_well_keys(frame)) == ["plate3_r1_c1"]

    def test_and_a_plate_column_is_used_when_it_is_there(self):
        frame = pd.DataFrame({"plateID": ["plate9"], "rowID": ["r1"],
                              "columnID": ["c1"]})
        frame.attrs["spacr_scan_plate"] = "plate3"

        assert list(_well_keys(frame)) == ["plate9_r1_c1"]


class TestItStillSaysWhenItCannot:

    def test_a_table_with_no_well_columns_says_so(self, tmp_path):
        path = tmp_path / "bare.csv"
        pd.DataFrame({"grna_name": ["TGGT1_000000_1"],
                      "count": [10]}).to_csv(path, index=False)

        got = regression_design_scan({"count_data": [str(path)]})

        assert "were not counted" in got["note"]
        # `None`, not 0: the scan distinguishes "counted, and there were
        # none" from "could not count", and the console prints a different
        # sentence for each.
        assert got["wells"] is None

    def test_no_count_files_at_all(self):
        assert "no count files" in regression_design_scan({})["note"]

    def test_an_unreadable_file_is_named(self, tmp_path):
        got = regression_design_scan(
            {"count_data": [str(tmp_path / "gone.csv")]})

        assert "could not read" in got["note"]


@pytest.mark.slow
def test_the_example_screen_reports_its_real_size():
    """The reported case, on the tables it was reported against."""
    from spacr.example_data import cache_folder

    files = sorted(glob.glob(f"{cache_folder()}/*unique_combinations.csv"))
    if len(files) != 4:
        pytest.skip("the example screen is not downloaded")

    got = regression_design_scan({"count_data": files})

    assert got["genes"] == 452
    assert got["guides"] == 1380
    assert got["wells"] == 1536, "384 wells is one plate's worth of four"
    assert got["note"] == ""
