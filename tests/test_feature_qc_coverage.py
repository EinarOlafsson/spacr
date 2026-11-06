"""Database paths of the feature QC in spacr.sp_stats: empty and clean
tables, and a flagged column SQLite refuses to drop."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from spacr import sp_stats as s


def _write(db, table, frame):
    with sqlite3.connect(db) as con:
        frame.to_sql(table, con, index=False, if_exists="replace")


def _ids(n):
    return pd.DataFrame({"plateID": "p1", "rowID": "r1",
                         "columnID": [f"c{i % 4 + 1}" for i in range(n)],
                         "object_label": np.arange(1, n + 1)})


def test_an_empty_and_a_clean_table_flag_nothing_and_write_an_empty_report(tmp_path):
    db = str(tmp_path / "measurements.db")
    rng = np.random.default_rng(0)
    n = 80
    # cell has rows and only well-behaved features; nucleus has no rows
    _write(db, "cell", _ids(n).assign(cell_area=rng.normal(100, 10, n),
                                      cell_eccentricity=rng.beta(2, 5, n)))
    _write(db, "nucleus", _ids(0).assign(nucleus_area=pd.Series(dtype=float)))

    flagged = s._feature_qc_database(db)
    assert flagged.empty
    assert list(flagged.columns) == list(s._FEATURE_QC_COLUMNS)

    flagged, written, dropped = s._feature_qc_report(db, drop=True)
    assert flagged.empty and dropped == []
    summary = json.loads(open(written["json"], encoding="utf-8").read())
    assert summary["flagged"] == 0 and summary["dropped"] == []
    assert summary["drop"] is True
    assert set(summary["reasons"].values()) == {0}
    report = pd.read_csv(written["csv"])
    assert report.empty and list(report.columns) == list(s._FEATURE_QC_COLUMNS)


def test_a_flagged_column_in_an_index_is_kept_and_left_out_of_the_dropped_list(tmp_path):
    db = str(tmp_path / "measurements.db")
    rng = np.random.default_rng(1)
    n = 80
    _write(db, "cell", _ids(n).assign(cell_area=rng.normal(100, 10, n),
                                      cell_flat=1.0, cell_also_flat=2.0))
    with sqlite3.connect(db) as con:
        con.execute('CREATE INDEX idx_flat ON cell ("cell_flat")')

    flagged, _, dropped = s._feature_qc_report(db, drop=True)
    assert set(flagged["feature"]) == {"cell_flat", "cell_also_flat"}
    assert set(flagged["reason"]) == {"constant"}
    assert dropped == ["cell.cell_also_flat"]
    with sqlite3.connect(db) as con:
        columns = {row[1] for row in con.execute('PRAGMA table_info("cell")')}
    assert "cell_flat" in columns and "cell_also_flat" not in columns
    assert "cell_area" in columns
