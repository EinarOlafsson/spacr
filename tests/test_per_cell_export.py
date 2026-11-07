"""One row per cell joining cell, cytoplasm and nucleus (item 769).

The BBBC021 reanalysis built this join in its own driver: cytoplasm 1:1,
each cell's largest nucleus, Metadata_n_nuclei, NaN nucleus columns for the
cells without one (3-8 % of cells had several nuclei, 4-8 % none). On two
of its plates spacr.sp_stats.per_cell_table reproduces the driver's parquet
row for row in all 571 shared columns (2026-11-07); these tests pin the
same rules on a small database.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr.tabular import read_table, write_database


def _keys(prcf, labels):
    plate, row, column, field = prcf.split("_")
    return {"plateID": plate, "rowID": row, "columnID": column,
            "fieldID": field, "prcf": prcf, "object_label": labels}


def _plate_db(path):
    """Two fields: multi-nucleus, nucleus-less and stray-nucleus cases."""
    cells = pd.concat([
        pd.DataFrame({**_keys("plate1_r2_c3_f1", [1, 2, 3]),
                      "cell_area": [900.0, 1200.0, 800.0],
                      "cell_centroid-0": [10.0, 20.0, 30.0],
                      "cell_distance_to_field_edge": [5.0, 6.0, 7.0],
                      "cell_zernike_0": [0.25, 0.25, 0.25]}),
        pd.DataFrame({**_keys("plate1_r2_c3_f2", [1, 2]),
                      "cell_area": [700.0, 1000.0],
                      "cell_centroid-0": [40.0, 50.0],
                      "cell_distance_to_field_edge": [8.0, 9.0],
                      "cell_zernike_0": [0.25, 0.25]}),
    ], ignore_index=True)
    cytoplasm = cells.loc[:, ["plateID", "rowID", "columnID", "fieldID",
                              "prcf", "object_label"]].copy()
    cytoplasm["cytoplasm_area"] = cells["cell_area"] - 300
    cytoplasm = cytoplasm.iloc[::-1].reset_index(drop=True)
    nuclei = pd.concat([
        pd.DataFrame({**_keys("plate1_r2_c3_f1", [1, 2, 3, 4]),
                      "cell_id": [1.0, 1.0, 2.0, np.nan],
                      "nucleus_area": [200.0, 300.0, 250.0, 90.0],
                      "nucleus_channel_0_mean_intensity": [10.0, 20.0, 30.0,
                                                           40.0]}),
        pd.DataFrame({**_keys("plate1_r2_c3_f2", [1]),
                      "cell_id": [1.0], "nucleus_area": [150.0],
                      "nucleus_channel_0_mean_intensity": [50.0]}),
    ], ignore_index=True)
    for table, frame in (("cell", cells), ("cytoplasm", cytoplasm),
                         ("nucleus", nuclei)):
        write_database(frame, path, table, if_exists="replace")
    return path


def test_one_row_per_cell_with_its_largest_nucleus(tmp_path):
    table = s.per_cell_table(_plate_db(str(tmp_path / "measurements.db")))
    assert list(table.columns[:6]) == [
        "Metadata_Plate", "Metadata_Well", "Metadata_Site",
        "Metadata_cell_id", "Metadata_n_nuclei", "cell_area"]
    assert table["Metadata_Well"].tolist() == ["B03"] * 5
    assert table["Metadata_Site"].tolist() == [1, 1, 1, 2, 2]
    assert table["Metadata_cell_id"].tolist() == [1, 2, 3, 1, 2]
    assert table["Metadata_n_nuclei"].tolist() == [2, 1, 0, 1, 0]
    assert table["cell_area"].tolist() == [900, 1200, 800, 700, 1000]
    assert table["cytoplasm_area"].tolist() == [600, 900, 500, 400, 700]
    np.testing.assert_array_equal(table["nucleus_area"],
                                  [300, 250, np.nan, 150, np.nan])
    np.testing.assert_array_equal(table["nucleus_channel_0_mean_intensity"],
                                  [20, 30, np.nan, 50, np.nan])
    assert table["cell_area"].dtype == np.float32
    assert "Location_cell_centroid-0" in table.columns
    assert "Location_cell_distance_to_field_edge" in table.columns
    assert not {"prcf", "object_label", "cell_id", "nucleus_cell_id",
                "cell_centroid-0"} & set(table.columns)
    assert table.attrs["per_cell"] == {
        "n_cells": 5, "nucleus": "largest",
        "tables": ["cell", "cytoplasm", "nucleus"], "n_nuclei": 5,
        "nuclei_without_cell": 1, "cells_without_nucleus": 2,
        "cells_multi_nucleus": 1}


@pytest.mark.parametrize("nucleus, area, intensity", [
    ("mean", 250.0, 15.0), ("median", 250.0, 15.0)])
def test_several_nuclei_can_be_aggregated(tmp_path, nucleus, area, intensity):
    table = s.per_cell_table(_plate_db(str(tmp_path / "measurements.db")),
                             nucleus=nucleus, dtype=None)
    assert table.loc[0, "nucleus_area"] == area
    assert table.loc[0, "nucleus_channel_0_mean_intensity"] == intensity
    assert table.loc[1, "nucleus_area"] == 250.0
    assert np.isnan(table.loc[2, "nucleus_area"])
    assert table["Metadata_n_nuclei"].tolist() == [2, 1, 0, 1, 0]
    assert table["cell_area"].dtype == np.float64


def test_export_writes_parquet_csv_feature_qc_and_counts(tmp_path):
    db = _plate_db(str(tmp_path / "measurements.db"))
    written = s.export_per_cell(db)
    assert set(written) == {"feature_qc", "parquet", "csv", "summary"}
    table = s.per_cell_table(db)
    for kind in ("parquet", "csv"):
        assert written[kind] == str(tmp_path / f"per_cell.{kind}")
        back = read_table(written[kind], canonicalise=False, report=None)
        assert list(back.columns) == list(table.columns)
        np.testing.assert_allclose(back["nucleus_area"],
                                   table["nucleus_area"], equal_nan=True)
    qc = read_table(written["feature_qc"], canonicalise=False, report=None)
    assert dict(zip(qc["feature"], qc["reason"])) == {
        "cell_zernike_0": "constant"}
    summary = json.loads((tmp_path / "per_cell.json").read_text(
        encoding="utf-8"))
    assert summary["n_rows"] == 5 and summary["cells_without_nucleus"] == 2
    assert summary["n_features"] == 5 and summary["flagged"] == 1
    assert summary["dropped"] == [] and summary["skipped"] == []
    assert summary["options"]["nucleus"] == "largest"


def test_export_can_leave_the_flagged_columns_out(tmp_path):
    db = _plate_db(str(tmp_path / "measurements.db"))
    out = tmp_path / "export"
    written = s.export_per_cell(db, str(out), name="cells", formats=["csv"],
                                drop_flagged=True)
    assert set(written) == {"feature_qc", "csv", "summary"}
    back = read_table(written["csv"], canonicalise=False, report=None)
    assert "cell_zernike_0" not in back.columns and len(back) == 5
    summary = json.loads((out / "cells.json").read_text(encoding="utf-8"))
    assert summary["dropped"] == ["cell_zernike_0"]


def test_a_cell_table_alone_still_exports(tmp_path):
    path = str(tmp_path / "measurements.db")
    write_database(pd.DataFrame({**_keys("p1_r1_c1_f1", [1, 2]),
                                 "cell_area": [1.0, 2.0]}), path, "cell")
    table = s.per_cell_table(path)
    assert list(table.columns) == ["Metadata_Plate", "Metadata_Well",
                                   "Metadata_Site", "Metadata_cell_id",
                                   "cell_area"]


def test_refusals_name_the_problem(tmp_path):
    db = _plate_db(str(tmp_path / "measurements.db"))
    with pytest.raises(ValueError, match="unknown nucleus choice"):
        s.per_cell_table(db, nucleus="smallest")
    with pytest.raises(ValueError, match="unknown format"):
        s.export_per_cell(db, formats=["xlsx"])
    empty = str(tmp_path / "nocells.db")
    write_database(pd.DataFrame({"x": [1]}), empty, "nucleus")
    with pytest.raises(ValueError, match="no cell table"):
        s.per_cell_table(empty)
    twice = str(tmp_path / "twice.db")
    write_database(pd.DataFrame({**_keys("p1_r1_c1_f1", [1, 1]),
                                 "cell_area": [1.0, 2.0]}), twice, "cell")
    with pytest.raises(ValueError, match="two rows for one cell"):
        s.per_cell_table(twice)


def test_measure_writes_the_export_only_when_asked(tmp_path, capsys):
    from spacr.measure import _emit_per_cell_export
    from spacr.settings import get_measure_crop_settings

    db = _plate_db(str(tmp_path / "measurements.db"))
    settings = get_measure_crop_settings({"src": str(tmp_path / "merged")})
    assert settings["per_cell_export"] is False
    assert settings["per_cell_nucleus"] == "largest"
    _emit_per_cell_export({**settings, "per_cell_nucleus": "mean",
                           "feature_qc_drop": True}, db)
    printed = capsys.readouterr().out
    assert ("Per-cell export: 5 cells, 4 features, 2 without a nucleus, "
            "1 with more than one") in printed
    back = read_table(tmp_path / "per_cell.parquet", canonicalise=False,
                      report=None)
    assert back.loc[0, "nucleus_area"] == 250.0
    assert "cell_zernike_0" not in back.columns


def test_the_export_never_fails_a_finished_run(tmp_path, capsys):
    from spacr.measure import _emit_per_cell_export

    broken = tmp_path / "measurements.db"
    broken.write_bytes(b"not a database")
    _emit_per_cell_export({}, str(broken))
    assert "Per-cell export could not run" in capsys.readouterr().out
    _emit_per_cell_export({}, str(tmp_path / "absent.db"))


def test_the_settings_are_alpha_gated():
    from spacr.settings import ALPHA_FEATURES, categories

    assert ALPHA_FEATURES[769]["settings"] == ("per_cell_export",
                                               "per_cell_nucleus")
    assert {"per_cell_export", "per_cell_nucleus"} <= set(
        categories["Profiling α"])
