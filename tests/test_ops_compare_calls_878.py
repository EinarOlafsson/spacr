"""Item 878: spaCR's per-nucleus genotype calls against other pipelines' tables.

Matching, categories and per-category decoy FDRs on synthetic tables, and the
compare_calls phase end to end on a two-field synthetic plate. Validation
against the decode-yield study's three-way comparison is recorded in the
item's feature file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from spacr import ops_decode, ops_engine
from spacr.ops_store import write_table
from spacr.settings import ALPHA_FEATURES
from spacr.tabular import write_table as write_file


def _table(points):
    return pd.DataFrame({"cell": range(1, len(points) + 1),
                         "y": [p[0] for p in points], "x": [p[1] for p in points]})


def test_units_are_mutual_neighbours_with_one_cell_per_table():
    tables = {"S": _table([(0, 0), (50, 50), (100, 100)]),
              "B": _table([(1, 0), (50, 53), (300, 300)]),
              "F": _table([(0, 1), (52, 50), (101, 100.5)])}
    units = ops_decode.match_units(tables, max_px=4.0)
    as_sets = sorted(tuple(sorted(u.items())) for u in units)
    assert (("B", 0), ("F", 0), ("S", 0)) in as_sets
    assert (("B", 1), ("F", 1), ("S", 1)) in as_sets
    assert (("F", 2), ("S", 2)) in as_sets
    assert (("B", 2),) in as_sets
    assert len(units) == 4
    assert ops_decode.match_units({"S": _table([]), "B": _table([(0, 0)])}) == [{"B": 0}]


@pytest.mark.parametrize("calls, expected", [
    ({"S": "A", "B": "A", "F": "A"}, ("all3_same", "")),
    ({"S": "A", "B": "B", "F": "C"}, ("all3_diff", "")),
    ({"S": "A", "B": "A", "F": "C"}, ("two_agree_third_diff", "F differs")),
    ({"S": "A", "B": "A", "F": None}, ("two_agree_third_none", "S+B agree, F none")),
    ({"S": "A", "B": "C", "F": None}, ("two_disagree_third_none", "S+B differ, F none")),
    ({"S": "A", "B": None, "F": None}, ("one_only", "S-only")),
    ({"S": None, "B": None, "F": None}, ("none", "")),
    ({"S": "A", "B": "A"}, ("pair_agree", "S+B")),
    ({"S": "A", "B": "C"}, ("pair_disagree", "S+B")),
])
def test_every_agreement_category(calls, expected):
    assert ops_decode.call_category(calls) == expected


def test_categories_carry_a_decoy_fdr_per_table():
    tables = {"S": _table([(0, 0), (50, 50), (100, 100)]),
              "B": _table([(0, 1), (50, 51), (100, 101)])}
    calls = {"S": {1: "GUIDE", 2: "DECOY", 3: "GUIDE"}, "B": {1: "GUIDE", 3: "OTHER"}}
    rows = ops_decode.compare_field(tables, calls, {"DECOY"})
    for row in rows:
        row.update({"plate": "P", "field": "P_A1_0"})
    units = pd.DataFrame(rows)
    assert sorted(units["category"]) == ["one_only", "pair_agree", "pair_disagree"]
    assert set(units["present"]) == {"SB"}
    categories, per_table = ops_decode.compare_summary(units, ["S", "B"], decoy_k=4,
                                                       fdr_scale=1.268)
    one = categories[(categories["plate"] == "P") & (categories["category"] == "one_only | S-only")]
    assert int(one["S_decoy_calls"].iloc[0]) == 1 and int(one["S_real_calls"].iloc[0]) == 0
    agree = per_table[(per_table["plate"] == "ALL") & (per_table["table"] == "S")].iloc[0]
    assert agree["real_calls"] == 2 and agree["decoy_calls"] == 1
    assert agree["decoy_FDR_pct"] == pytest.approx(round(100 * 1.268 / 4 * 1 / 2, 3))
    assert agree["assigned_pct"] == pytest.approx(66.67)


def test_the_region_keeps_owned_cells_clear_of_the_border():
    frame = _table([(5, 50), (50, 50), (50, 95), (50, 60)])
    centres = np.array([[50.0, 50.0], [50.0, 150.0]])
    kept = ops_decode.compare_region(frame, 0, (0.0, 0.0), centres, [0, 1],
                                     tile=(100, 100), edge=10)
    assert kept["cell"].tolist() == [2, 4]
    assert ops_decode.compare_region(frame.iloc[:0], 0, (0.0, 0.0), centres, [0, 1],
                                     tile=(100, 100)).empty


def test_the_tables_setting_takes_pairs_json_or_a_dict():
    assert ops_decode._compare_tables_setting("a=/x.csv, b=/y.csv") == {"a": "/x.csv", "b": "/y.csv"}
    assert ops_decode._compare_tables_setting('{"a": "/x.csv"}') == {"a": "/x.csv"}
    assert ops_decode._compare_tables_setting({"a": 1}) == {"a": "1"}
    with pytest.raises(ValueError, match="ops_compare_tables"):
        ops_decode._compare_tables_setting("")


@pytest.fixture()
def plate(tmp_path):
    """A two-site well with spaCR calls and one external table."""
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_geometry", pd.DataFrame({
        "plate": "P", "well": "A1", "site": [0, 1], "cycle": 1, "y": [0.0, 0.0],
        "x": [0.0, 100.0], "tile_height": 100, "tile_width": 100}))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": "P", "well": "A1", "object_id": [1, 2, 3, 4],
        "centroid_y": [50.0, 30.0, 70.0, 50.0], "centroid_x": [50.0, 30.0, 70.0, 150.0],
        "area": 50.0}))
    calls = pd.DataFrame({"well": "A1", "site": [0, 0, 1], "object_id": [1, 2, 4],
                          "mapped_guide": ["GUIDE1", "GUIDE2", "GUIDE3"],
                          "is_decoy": False})
    write_file(calls, os.path.join(ops_decode.results_folder(db),
                                   "calls_A1.parquet"), canonicalise=False)
    other = pd.DataFrame({"well": "A1", "site": [0, 0, 0, 1], "y": [50.5, 30.0, 70.0, 50.0],
                          "x": [50.0, 31.0, 85.0, 50.0],
                          "guide": ["GUIDE1", "GUIDE9", None, "GUIDE3"]})
    path = str(tmp_path / "other.csv")
    other.to_csv(path, index=False)
    return db, path


def test_the_compare_phase_writes_units_categories_and_tables(plate):
    db, path = plate
    report = ops_decode.run_compare_calls(
        db, "P", ["A1"], {"ops_compare_tables": {"X": path},
                          "ops_decoy_library_k": 4}, ["GUIDE1", "GUIDE2", "GUIDE3"])
    assert report["units"] > 0 and report["fields"] == 2
    units = pd.read_parquet(report["paths"]["units"])
    pair = units[units["spaCR_cell"] == 1].iloc[0]
    assert pair["category"] == "pair_agree"
    assert units[units["spaCR_cell"] == 2]["category"].iloc[0] == "pair_disagree"
    assert units[units["spaCR_cell"] == 4]["category"].iloc[0] == "pair_agree"
    assert (units["present"] == "X").sum() == 1
    categories = pd.read_csv(report["paths"]["categories"])
    assert {"plate", "category", "present", "n", "spaCR_real_calls",
            "X_decoy_FDR_pct"} <= set(categories.columns)
    tables = pd.read_csv(report["paths"]["tables"])
    assert set(tables["table"]) == {"spaCR", "X"}


def test_the_compare_phase_refuses_what_it_cannot_read(plate, tmp_path):
    db, path = plate
    bad = str(tmp_path / "bad.csv")
    pd.DataFrame({"well": ["A1"], "x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        ops_decode.run_compare_calls(db, "P", ["A1"], {"ops_compare_tables": {"X": bad}},
                                     ["GUIDE1"])
    with pytest.raises(ValueError, match="calls_B2"):
        ops_decode.run_compare_calls(db, "P", ["A1"], {
            "ops_compare_tables": {"X": path}, "ops_compare_fields": "B2:0"}, ["GUIDE1"])
    report = ops_decode.run_compare_calls(db, "P", ["A1"], {
        "ops_compare_tables": {"X": path}, "ops_compare_fields": ["A1:7"]}, ["GUIDE1"])
    assert report == {"units": 0, "fields": 1}


def test_the_compare_phase_runs_from_run_ops(plate):
    db, path = plate
    out = ops_engine.run_ops({"dst_root": os.path.dirname(db), "plate": "P",
                              "ops_compare_tables": f"X={path}"},
                             wells=["a1"], phases=("compare_calls",),
                             library=["GUIDE1", "GUIDE2", "GUIDE3"])
    assert out["compare_calls"]["units"] > 0
    with open(os.path.join(ops_decode.results_folder(db, "ops_compare"),
                           "compare_calls_P.json"), encoding="utf-8") as handle:
        assert json.load(handle)["units"] == out["compare_calls"]["units"]


def test_the_compare_settings_are_alpha():
    from spacr import ops_settings

    for key in ALPHA_FEATURES[878]["settings"]:
        assert key in ops_settings.OPS_DEFAULTS and "Alpha." in ops_settings.OPS_TOOLTIPS[key]
