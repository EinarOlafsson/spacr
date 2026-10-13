"""png_list is not written the same way on every plate, and the montage joins.

Reported 2026-08-19 as an empty montage that explained itself precisely:

    wells: 0 of 8 wells reporting the guide contributed an object (none)
    plate1_r3_c24: round(0 x 0.1817) = 0 -> 0 shown
      (no object in the imported databases comes from this well)

while the baseline was computed over all 226,467 objects -- so the objects
were loaded and simply did not join. Measured on those four databases:

    plate1  rowID / columnID   and plateID = 'pplate1'
    plate2  row_name / column_name
    plate3  row_name / column_name
    plate4  row_name / column_name

Plates 2-4 could not compose a `prc` at all, and plate1 composed one against a
doubled plate name matching nothing in the counts. Instruction 145: the reader
that does not canonicalise is the reader that silently disagrees.
"""
import sqlite3

import pandas as pd
import pytest

from spacr.cell_montage import load_montage_objects


def _plate(tmp_path, name, *, row_key, column_key, plate_value):
    plate = tmp_path / name
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    crops.mkdir(parents=True)
    rows = []
    for i in range(4):
        crop = crops / f"{name}_A0{i}.png"
        crop.write_bytes(b"x")
        rows.append({"png_path": str(crop), "file_name": crop.name,
                     "plateID": plate_value, row_key: "r3",
                     column_key: "c24", "fieldID": "f1", "cell_id": i,
                     "pred": 0.1 * i})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)
    return str(db)


def test_the_canonical_spelling_joins(tmp_path):
    db = _plate(tmp_path, "plate1", row_key="rowID", column_key="columnID",
                plate_value="plate1")

    frame = load_montage_objects(db)

    assert set(frame["prc"]) == {"plate1_r3_c24"}


def test_the_old_spelling_joins_too(tmp_path):
    """row_name / column_name is what three of the four plates carry."""
    db = _plate(tmp_path, "plate2", row_key="row_name",
                column_key="column_name", plate_value="plate2")

    frame = load_montage_objects(db)

    assert "prc" in frame.columns, "no well key could be composed at all"
    assert set(frame["prc"]) == {"plate2_r3_c24"}


def test_a_doubled_plate_name_is_normalised(tmp_path):
    """`pplate1` is what plate1's png_list actually holds."""
    db = _plate(tmp_path, "plate1", row_key="rowID", column_key="columnID",
                plate_value="pplate1")

    frame = load_montage_objects(db)

    assert set(frame["prc"]) == {"plate1_r3_c24"}, (
        "the doubled plate name reached the well key and matched no count row")


def test_a_route_that_is_not_here_is_not_reported_as_failures():
    """A screen with PNG crops and no merged/ folder is healthy."""
    from spacr.crops import ReanchorReport

    absent = ReanchorReport(root="/screen", n_paths=60816, n_reanchored=0,
                            n_already=0, failures=("/old/x.npy",) * 3)
    assert "not on this machine" in absent.describe()
    assert "could not be re-anchored" not in absent.describe()

    partial = ReanchorReport(root="/screen", n_paths=100, n_reanchored=97,
                             n_already=0, failures=("/old/x.png",))
    assert "could not be re-anchored" in partial.describe()
