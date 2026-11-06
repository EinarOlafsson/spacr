"""A plate name stored URL-escaped keeps its rows through the merge (item 805).

A PerkinElmer export stored ``AssayPlate%5FPerkinElmer`` in ``prcf`` while the
plateID was decoded, so the inner joins matched nothing and every row dropped.
"""
import sqlite3

import pandas as pd
import pytest

from spacr import io as IO
from spacr import schema

PLATE = "AssayPlate_PerkinElmer"


def _frame(prcf_plate, extra):
    rows = []
    for obj in range(1, 5):
        rows.append({
            "plateID": PLATE, "rowID": "r1", "columnID": "c1", "fieldID": "f1",
            "prcf": f"{prcf_plate}_r1_c1_f1", "object_label": obj,
            "cell_id": obj, **{k: v + obj for k, v in extra.items()}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("stored", [
    "AssayPlate%5FPerkinElmer", "AssayPlate%255FPerkinElmer", PLATE])
def test_escaped_plate_names_decode_to_one_spelling(stored):
    assert schema.canonical_plate_id(stored) == PLATE


def test_rows_survive_the_merge_when_prcf_is_url_escaped(tmp_path):
    db = tmp_path / "m.db"
    png = _frame("AssayPlate%255FPerkinElmer", {}).assign(
        cell_id=lambda f: "o" + f["object_label"].astype(str),
        png_path=lambda f: "/x/cell_png/o" + f["object_label"].astype(str) + ".png",
        file_name=lambda f: "o" + f["object_label"].astype(str) + ".png")
    with sqlite3.connect(db) as con:
        _frame("AssayPlate%5FPerkinElmer", {"cell_area": 10.0}).assign(
            measurement_ndim=2, n_z=1).drop(columns="cell_id").to_sql("cell", con, index=False)
        png.to_sql("png_list", con, index=False)

    merged, _ = IO._read_and_merge_data([str(db)], ["cell", "png_list"])

    assert len(merged) == 4
    assert "cell_area" in merged.columns
    assert merged["cell_area"].notna().all()
    assert merged["measurement_ndim"].notna().all()
    assert merged["png_path"].notna().all()
    assert merged["png_path"].str.endswith(".png").all()
    assert all("%" not in str(i) for i in merged.index)
