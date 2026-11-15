"""Table-driven tests of the readers that consume a ``measurements.db`` written by ``spacr.tabular.write_database``.

The per-cell export (``spacr.sp_stats.per_cell_table`` / ``export_per_cell``)
is walked over the join keys, nucleus choices and timelapse metadata it needs
and the refusals it gives without them; the timelapse bleach correction
(``spacr.timelapse._correct_timelapse_bleaching``) skips tables with no
timepoint and draws nothing for an empty fit.  Every store is a real SQLite
file in ``tmp_path``.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr import timelapse as tl
from spacr.tabular import write_database
from tests.test_bleach_correction import _bleaching_table


def _keys(prcf, labels, time=None):
    """The join-key columns of one field's objects."""
    plate, row, column, field = prcf.split("_")
    keys = {"plateID": plate, "rowID": row, "columnID": column,
            "fieldID": field, "prcf": prcf, "object_label": labels}
    if time is not None:
        keys["timeID"] = time
    return keys


def _store(path, tables):
    """Write ``{name: frame}`` into one SQLite store and return its path."""
    for name, frame in tables.items():
        write_database(frame, path, name)
    return path


def test_per_cell_table_joins_its_compartments_or_refuses_by_name(
        tmp_path, subtests):
    """Missing keys, a duplicated cytoplasm row and a nucleus without area are refused; time is kept."""
    cell = pd.DataFrame({**_keys("p1_r1_c1_f1", [1, 2]),
                         "cell_area": [1.0, 2.0]})
    refused = [
        ("no-object-label", {"cell": cell.drop(columns=["object_label"])},
         {}, "no object_label column to join on"),
        ("cytoplasm-twice", {"cell": cell, "cytoplasm": pd.DataFrame(
            {**_keys("p1_r1_c1_f1", [1, 1]), "cytoplasm_area": [1.0, 2.0]})},
         {}, "cytoplasm table has two rows"),
        ("largest-nucleus-needs-area", {
            "cell": pd.DataFrame({**_keys("p1_r1_c1_f1", [1]),
                                  "cell_area": [1.0]}),
            "nucleus": pd.DataFrame({**_keys("p1_r1_c1_f1", [1]),
                                     "cell_id": [1.0],
                                     "nucleus_mean": [3.0]})},
         {}, "no nucleus_area"),
    ]
    for case_id, tables, kwargs, fragment in refused:
        with subtests.test(msg=case_id):
            path = _store(str(tmp_path / f"{case_id}.db"), tables)
            with pytest.raises(ValueError, match=fragment):
                s.per_cell_table(path, **kwargs)

    with subtests.test(msg="nucleus-by-mean"):
        path = str(tmp_path / "largest-nucleus-needs-area.db")
        table = s.per_cell_table(path, nucleus="mean")
        assert table["nucleus_mean"].tolist() == [3.0]
        assert table["cell_area"].tolist() == [1.0]

    timed = pd.DataFrame({**_keys("p1_r1_c1_f1", [1, 2], time=[0, 1]),
                          "cell_area": [1.0, 2.0]})
    with subtests.test(msg="timelapse-keeps-time"):
        table = s.per_cell_table(_store(str(tmp_path / "t.db"),
                                        {"cell": timed}))
        assert table["Metadata_timeID"].tolist() == [0, 1]
        assert table["cell_area"].tolist() == [1.0, 2.0]

    with subtests.test(msg="no-plate-or-well-columns"):
        bare = timed.loc[:, ["prcf", "object_label", "cell_area"]]
        table = s.per_cell_table(_store(str(tmp_path / "bare.db"),
                                        {"cell": bare}))
        assert list(table.columns) == ["Metadata_cell_id", "cell_area"]
        assert table["cell_area"].tolist() == [1.0, 2.0]


def test_export_per_cell_lists_a_format_whose_writer_is_missing(
        tmp_path, monkeypatch):
    """A writer that raises ImportError is reported as skipped; the other formats are still written."""
    import spacr.tabular as tabular

    real = tabular.write_table

    def fake(frame, path, **kwargs):
        if str(path).endswith(".parquet"):
            raise ImportError("pyarrow is missing")
        return real(frame, path, **kwargs)

    monkeypatch.setattr(tabular, "write_table", fake)
    path = _store(str(tmp_path / "measurements.db"), {"cell": pd.DataFrame(
        {**_keys("p1_r1_c1_f1", [1, 2]), "cell_area": [1.0, 2.0]})})
    written = s.export_per_cell(path)
    assert "parquet" not in written
    assert "csv" in written
    assert pd.read_csv(written["csv"])["cell_area"].tolist() == [1.0, 2.0]
    with open(written["summary"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["skipped"] == ["parquet: pyarrow is missing"]


def test_timelapse_bleaching_skips_tables_without_time_and_draws_no_empty_fit(
        tmp_path, monkeypatch, subtests):
    """A table without timepoints is skipped, an empty fit draws nothing, and no timed table is refused."""
    with subtests.test(msg="empty-fit-draws-nothing"):
        db = tmp_path / "plate1" / "measurements" / "measurements.db"
        write_database(_bleaching_table(n_frames=4, n_objects=5), db, "cell",
                       if_exists="replace", canonicalise=False)
        write_database(pd.DataFrame({"nucleus_channel_0_mean_intensity":
                                     [1.0]}),
                       db, "nucleus", if_exists="replace", canonicalise=False)
        real = tl._bleach_correct_table
        seen = []

        def no_fits(df, table, method):
            seen.append(table)
            corrected, fits = real(df, table, method)
            return corrected, fits.iloc[0:0]

        monkeypatch.setattr(tl, "_bleach_correct_table", no_fits)
        fits = tl._correct_timelapse_bleaching(str(db), "ratio", plot=True)
        monkeypatch.setattr(tl, "_bleach_correct_table", real)
        assert seen == ["cell"]
        assert fits.empty
        assert not (tmp_path / "plate1" / "results" /
                    "bleach_correction").exists()
        with sqlite3.connect(db) as conn:
            names = sorted(r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"))
        assert names == ["bleach_correction", "cell",
                         "cell_bleach_corrected", "nucleus"]

    with subtests.test(msg="no-timed-table-is-refused"):
        db = tmp_path / "untimed" / "measurements" / "measurements.db"
        write_database(pd.DataFrame({"nucleus_channel_0_mean_intensity":
                                     [1.0]}),
                       db, "nucleus", if_exists="replace", canonicalise=False)
        with pytest.raises(ValueError) as refused:
            tl._correct_timelapse_bleaching(str(db), "ratio", plot=False)
        assert str(refused.value) == (
            f"{db} has no timelapse object table with channel intensities to "
            f"correct")
