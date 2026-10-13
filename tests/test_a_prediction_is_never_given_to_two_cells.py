"""Item 288: Investigate Hit refuses a join that would score a cell twice.

``spacr.hit_investigation._read_cells`` joins a prediction file to the cells
a Measure run wrote. The join is one prediction to one cell, and two ways of
breaking that are refused with a sentence that says which side repeats:

* the measured cells repeat an object key (two labels written under one
  ``prcfo``), so a prediction for that key cannot be given to one of them;
* the prediction file scores two crops that ``png_list`` records under the
  same object key once the plate id is made canonical (a legacy ``pplate1``
  crop beside a ``plate1`` one), so the object would carry two scores.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionError
from spacr.hit_investigation import _read_cells
from tests.test_hit_investigation_joins_a_measure_database import (
    _write_measure_db)


def test_cells_that_repeat_an_object_key_are_refused(tmp_path):
    database, _crops = _write_measure_db(tmp_path, cell_prcfo=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE cell SET prcfo = 'plate1_r1_c1_f1_o1' "
            "WHERE object_label = 2 AND columnID = 'c1'")
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.9]}).to_csv(
        predictions, index=False)
    with pytest.raises(HitAttributionError) as excinfo:
        _read_cells(str(database), str(predictions), "pred", "path")
    assert "repeat prcfo" in str(excinfo.value)


def test_two_scored_crops_of_one_object_are_refused(tmp_path):
    database, crops = _write_measure_db(tmp_path)
    first = crops.iloc[0]["png_path"]
    twin = first[: first.rindex("_")] + "_99.png"
    with sqlite3.connect(database) as connection:
        cols = [r[1] for r in connection.execute('PRAGMA table_info("png_list")')]
        row = dict(zip(cols, connection.execute(
            'SELECT * FROM png_list WHERE png_path = ?',
            (crops.iloc[0]["png_path"],)).fetchone()))
        row["png_path"] = twin
        row["prcfo"] = "p" + row["prcfo"]
        if "plateID" in row:
            row["plateID"] = "p" + row["plateID"]
        marks = ",".join("?" * len(cols))
        connection.execute(f'INSERT INTO png_list VALUES ({marks})',
                           [row[c] for c in cols])
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": [crops.iloc[0]["png_path"], twin],
                  "pred": [0.2, 0.8]}).to_csv(predictions, index=False)
    with pytest.raises(HitAttributionError) as excinfo:
        _read_cells(str(database), str(predictions), "pred", "path")
    assert "more than one crop of the same object" in str(excinfo.value)
