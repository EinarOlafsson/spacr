"""Coverage for the recruitment and replication filter, panel and report helpers of the analysis submodules."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import submodules as sm


def _conditions():
    return pd.DataFrame({
        "condition": ["a_x_p1", "a_x_p1", "b_x_p2", "b_x_p2"],
        "host_cells": ["a", "a", "b", "b"],
        "treatment": ["x", "x", "x", "x"],
        "pathogen": ["p1", "p1", "p2", "p2"],
        "plateID": ["plate1"] * 4,
        "file_name": ["f1", "f1", "f2", "f2"],
        "recruitment": [1.0, 2.0, 3.0, 4.0],
    })


def test_a_replicate_inside_the_condition_pools_the_other_slots():
    """When each condition holds one replicate the QC group drops that slot."""
    df = _conditions()
    group, frame = sm._qc_group_column(df, {"replicate_column": "pathogen"})
    assert group == "_qc_group"
    assert frame["_qc_group"].tolist() == ["a_x", "a_x", "b_x", "b_x"]
    assert sm._qc_group_column(df, {})[0] == "condition"
    assert sm._qc_group_column(df, {"qc_group": None}) == (None, df)
    assert sm._qc_group_column(df, {"qc_group": "plateID"})[0] == "plateID"
    assert sm._qc_group_column(df, {"qc_group": "nothing"})[0] is None


def test_each_qc_filter_runs_alone(monkeypatch):
    """Well QC alone and object QC alone each log only their own step."""
    from spacr import assay_report

    df = pd.DataFrame({"plateID": ["p"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
                       "columnID": ["c1", "c2", "c1", "c2"], "x": [1.0] * 4})

    def fake_wells(frame, channels, measures, mad, row_object, group=None, min_wells=8):
        return pd.DataFrame({
            "plateID": ["p", "p"], "rowID": ["r1", "r2"], "columnID": ["c1", "c1"],
            "well_qc_pass": [False, True], "qc_group": ["g", "g"],
            "qc_reference": ["plate (p)", "group"]})

    def fake_objects(frame, channels, measures, mad, group=None, min_objects=30):
        keep = np.array([True, True, False, True])
        summary = pd.DataFrame({"column": ["x"], "objects_removed": [1]})
        return keep, summary

    monkeypatch.setattr(assay_report, "well_qc_table", fake_wells)
    monkeypatch.setattr(assay_report, "object_qc_mask", fake_objects)

    kept, steps, tables = sm._qc_filters(df, {"well_qc": True})
    assert list(tables) == ["well_qc"] and len(kept) == 3
    assert [step[0] for step in steps] == ["well_qc (MAD 3.5)"]
    assert "1 of 2 wells failed" in steps[0][3] and "r1c1" in steps[0][3]

    kept, steps, tables = sm._qc_filters(df, {"object_qc": True})
    assert list(tables) == ["object_qc_summary"] and len(kept) == 3
    assert [step[0] for step in steps] == ["object_qc (MAD 3.5)"]
    assert steps[0][3].endswith("x (1)")


def test_max_vacuoles_needs_the_merge_column(tmp_path):
    """The vacuole limit refuses a table without the merged vacuole count."""
    df = _conditions()
    with pytest.raises(KeyError, match="pathogen_prcfo_count"):
        sm._recruitment_new_filters(df, {"max_vacuoles_per_cell": 2}, str(tmp_path))


def test_figures_naming_an_unknown_panel_are_refused(tmp_path, monkeypatch):
    """A figure that lists a panel id that does not exist stops the run before writing."""
    monkeypatch.setattr(sm, "_bar_panel_from_objects", lambda *a, **k: {"group": "condition"})
    settings = {"panels": [{"name": "one"}], "figures": {"fig": ["P01", "P99"]}}
    with pytest.raises(KeyError, match="P99"):
        sm._recruitment_panels_impl(_conditions(), settings, str(tmp_path))


def test_the_report_grouping_follows_labels_and_a_given_order():
    """Unknown replicate columns are ignored; labels and the condition order set the shown names."""
    df = pd.DataFrame({"condition": ["a", "a", "b", "c"], "recruitment": [1.0, 2.0, 3.0, 4.0]})
    settings = {"replicate_column": "missing", "condition_order": ["b", "a", "c"],
                "condition_labels": {"a": "A shown"}}
    group, order, shown, _add, rep = sm._report_grouping(df, settings)
    assert group == "condition" and order == ["b", "a", "c"]
    assert shown == {"a": "A shown"}
    assert rep is None


def test_the_tile_config_skips_channels_without_a_plane(tmp_path, monkeypatch):
    """Channels that are not set are left out of the colour map."""
    from spacr import assay_report

    monkeypatch.setattr(assay_report, "resolve_mask_dims",
                        lambda merged, names, planes: {n: i for i, n in enumerate(names)})
    merged = tmp_path / "merged"
    merged.mkdir()
    df = pd.DataFrame({"file_name": ["f1"]})
    settings = {"channel_of_interest": 2, "channel_dims": [0, 1, 2], "merged_dir": str(merged),
                "cell_chann_dim": 3, "nucleus_chann_dim": None, "pathogen_chann_dim": 1,
                "report_tiles_per_condition": 3, "report_tile_size": 32, "report_tile_seed": 1}
    cfg = sm._recruitment_tile_config(df, settings)
    assert cfg["colour_map"] == {2: "red", 1: "green"}
    assert cfg["mask_dims"] == {"cell": 0, "pathogen": 1}
    assert cfg["n"] == 3 and cfg["size"] == 32


def test_panel_crops_are_built_per_panel_and_a_failure_is_reported(tmp_path, monkeypatch, capsys):
    """Each user panel gets a crop grid; a panel whose grid fails is reported, not fatal."""
    from spacr import assay_report

    cfg = dict(merged_dir="m", mask_dims={}, colour_map={}, size=16, mono_channel=0)
    monkeypatch.setattr(sm, "_recruitment_tile_config", lambda df, settings: cfg)
    calls = []

    def fake_tiles(sub, group, order, merged, mask_dims, colours, folder, **kwargs):
        calls.append(os.path.basename(folder))
        if len(calls) == 2:
            raise RuntimeError("no tiles")
        os.makedirs(folder, exist_ok=True)
        return {"table": pd.DataFrame({"a": [1]})}

    monkeypatch.setattr(assay_report, "build_tiles", fake_tiles)
    df = _conditions()
    specs = {"P01": {"group": "condition", "value": "recruitment", "order": ["a_x_p1"]},
             "P02": {"group": "condition", "value": "missing_col", "order": ["b_x_p2"]}}
    names = {"P01": "first", "P02": "second"}
    out = sm._panel_crop_grids(df, specs, names, [{}, {}], str(tmp_path),
                               {"representative_crops": True, "representative_crops_n": 2})
    assert list(out) == ["P01"]
    assert len(calls) == 2
    assert os.path.isfile(os.path.join(calls[0] and str(tmp_path), calls[0], "data.csv"))
    assert "representative crops for P02 failed: no tiles" in capsys.readouterr().out


def test_the_filter_steps_need_a_log_with_counts():
    """No log, or a log that holds no counts, gives no per-group steps."""
    frame = _conditions()
    assert sm._report_filter_steps(None, frame, "condition") is None

    class Empty:
        def steps_by_group(self):
            return pd.DataFrame(columns=["step", "step_index", "condition", "remaining"])

    assert sm._report_filter_steps(Empty(), frame, "condition") is None


def test_the_recruitment_report_says_when_there_are_no_tiles(tmp_path, monkeypatch):
    """Without merged arrays the report introduction explains why there are no tiles."""
    from spacr import assay_report

    seen = {}

    def fake_report(frames, out_dir, value, group, order, channels, **kwargs):
        seen.update(kwargs, frames=frames, group=group, order=order)
        return {"pdf": "r.pdf", "n_pages": 1, "legend": "l.txt"}

    monkeypatch.setattr(assay_report, "build_assay_report", fake_report)
    settings = {"src": str(tmp_path), "channel_of_interest": 1, "channel_dims": [0, 1],
                "target": "t", "report_max_conditions_per_row": 4, "representative_crops_n": 2}
    res = sm._recruitment_report_impl(_conditions(), _conditions(), settings,
                                      str(tmp_path / "results"), None)
    assert res["n_pages"] == 1
    assert seen["tiles"] is None
    assert seen["intro"][1].startswith("No representative tiles or crops")
    assert seen["intro"][0].startswith("Recruitment report for")


def _merged_field(path, value=5.0, planes=3, size=16):
    arr = np.full((size, size, planes), value, dtype=np.float32)
    arr[..., -1] = 0
    arr[:4, :4, -1] = 1
    np.save(path, arr)


def test_field_backgrounds_are_cached_and_reused(tmp_path):
    """A second call reads the cache, a call for other channels recomputes, and a full cache reads nothing."""
    merged = tmp_path / "merged"
    merged.mkdir()
    _merged_field(merged / "f1.npy", 5.0)
    _merged_field(merged / "f2.npy", 7.0)
    cache = str(tmp_path / "results" / "bg.csv")
    first = sm._field_background(str(merged), ["f1"], [0], cache_path=cache, stride=1)
    assert first["ch0_field_background"].tolist() == [5.0]
    both = sm._field_background(str(merged), ["f1", "f2"], [0], cache_path=cache, stride=1)
    assert both["ch0_field_background"].tolist() == [5.0, 7.0]
    for name in ("f1", "f2"):
        os.remove(merged / f"{name}.npy")
    again = sm._field_background(str(merged), ["f1", "f2"], [0], cache_path=cache, stride=1)
    assert again["ch0_field_background"].tolist() == [5.0, 7.0]
    _merged_field(merged / "f1.npy", 9.0)
    other = sm._field_background(str(merged), ["f1"], [1], cache_path=cache, stride=1)
    assert other.columns.tolist() == ["file_name", "ch1_field_background"]
    assert other["ch1_field_background"].tolist() == [9.0]


def test_object_quality_filters_refuse_missing_columns(monkeypatch, tmp_path):
    """Background and solidity filters name the column they cannot find."""
    from spacr import measure

    log = sm._FilterLog(pd.DataFrame({"x": [1]}))
    rule = {"background_filters": [{"channel": 0, "min_ratio": 1}]}
    with pytest.raises(KeyError, match="'file_name' column"):
        sm._object_quality_filters(pd.DataFrame({"x": [1]}), rule, "pathogen", str(tmp_path), log, "r")

    monkeypatch.setattr(
        measure, "_image_table_background",
        lambda db, names, channels: pd.DataFrame(
            {f"channel_{c}_background_median": [2.0] * len(names) for c in channels}))
    frame = pd.DataFrame({"file_name": ["f1", "f2"], "pathogen_channel_1_mean_intensity": [1.0, 4.0]})
    with pytest.raises(KeyError, match="pathogen_channel_0_mean_intensity"):
        sm._object_quality_filters(frame, rule, "pathogen", str(tmp_path), log, "r")
    with pytest.raises(KeyError, match="pathogen_solidity"):
        sm._object_quality_filters(frame, {"min_solidity": 0.5}, "pathogen", str(tmp_path), log, "r")
    ok = sm._object_quality_filters(
        frame, {"background_filters": {"channel": 1, "min_ratio": 1}}, "pathogen", str(tmp_path),
        log, "r")
    assert ok["file_name"].tolist() == ["f2"]


def test_replication_filters_apply_the_solidity_filter(tmp_path):
    """The replication filters run the object quality filters from the project of the database."""
    frame = pd.DataFrame({"pathogen_solidity": [0.2, 0.9], "pathogen_area": [1.0, 2.0]})
    db = str(tmp_path / "measurements" / "measurements.db")
    kept, log, flags = sm._replication_filters(
        frame, {"min_solidity": 0.5}, "pathogen_area", lambda f: np.zeros(len(f), bool),
        [(db, "pathogen", None)], "objects", "start")
    assert kept["pathogen_solidity"].tolist() == [0.9]
    assert flags is None
    assert log.rows[-1]["step"] == "min_solidity 0.5"


def _tile_df():
    return pd.DataFrame({"file_name": ["f1"], "object_label": [1]})


def test_the_replication_tile_config_reads_the_layout_sidecar(tmp_path):
    """Missing or unreadable sidecars give a reason; colours follow the setting, the channel count or a default pair."""
    from spacr.crops import MERGED_LAYOUT_SIDECAR

    merged = tmp_path / "merged"
    merged.mkdir()
    base = {"src": str(tmp_path), "merged_dir": str(merged)}
    cfg, why = sm._replication_tile_config(_tile_df(), base)
    assert cfg is None and "no 'pathogen' mask plane" in why

    sidecar = merged / MERGED_LAYOUT_SIDECAR
    sidecar.write_text("not json")
    cfg, why = sm._replication_tile_config(_tile_df(), base)
    assert cfg is None

    sidecar.write_text(json.dumps({"mask_dims": {"pathogen": 3}}))
    cfg, why = sm._replication_tile_config(_tile_df(), dict(base, report_tile_channels={"1": "magenta"}))
    assert why == "" and cfg["colour_map"] == {1: "magenta"}
    cfg, _ = sm._replication_tile_config(_tile_df(), dict(base, channel_dims=[0, 1, 2]))
    assert cfg["colour_map"] == {2: "red", 1: "green", 0: "blue"}
    cfg, _ = sm._replication_tile_config(_tile_df(), dict(base, channel_dims=[0, 1]))
    assert cfg["colour_map"] == {0: "green", 1: "blue"}
    assert cfg["mask_dims"] == {"pathogen": 3}


def test_the_replication_report_groups_by_a_plain_column(tmp_path, monkeypatch):
    """A group column other than condition is used as it is, with the replicate dropped when unknown."""
    from spacr import assay_report

    seen = {}

    def fake_report(frames, out_dir, value, group, order, channels, **kwargs):
        seen.update(kwargs, group=group, order=order)
        return {"pdf": "r.pdf", "n_pages": 2}

    monkeypatch.setattr(assay_report, "build_assay_report", fake_report)
    df = _conditions()
    df["well"] = ["A1", "A1", "B1", "B1"]
    df["bin"] = ["lo", "hi", "lo", "hi"]
    settings = {"src": str(tmp_path), "replicate_column": "absent", "channel_dims": [0, 1],
                "compartment": "pathogen"}
    res = sm._replication_report_impl(
        settings, df, df, "well", "bin", ["lo", "hi"], "recruitment", "value", "legend",
        str(tmp_path / "out"), None, method="m", bin_name="bin")
    assert res["n_pages"] == 2
    assert seen["group"] == "well" and seen["order"] == ["A1", "B1"]
    assert seen["replicate_column"] is None


def test_recruitment_filters_apply_the_solidity_filter(tmp_path):
    """The post-merge filters run the object quality filters on the pathogen columns."""
    df = pd.DataFrame({"pathogen_solidity": [0.2, 0.9]})
    kept, log = sm._recruitment_new_filters(
        df, {"min_solidity": 0.5, "src": str(tmp_path)}, str(tmp_path))
    assert kept["pathogen_solidity"].tolist() == [0.9]
    assert log.rows[-1]["step"] == "min_solidity 0.5"


def test_panel_crops_are_returned_with_the_panels(tmp_path, monkeypatch):
    """Crop grids built for the panels come back under the crops key."""
    monkeypatch.setattr(sm, "_bar_panel_from_objects", lambda *a, **k: {"group": "condition"})
    monkeypatch.setattr(sm, "_write_panels_and_figures", lambda *a, **k: {"P01": "p.pdf"})
    monkeypatch.setattr(sm, "_panel_crop_grids", lambda *a, **k: {"P01": {"table": None}})
    out = sm._recruitment_panels_impl(_conditions(), {"panels": [{"name": "one"}]}, str(tmp_path))
    assert out == {"panels": {"P01": "p.pdf"}, "crops": {"P01": {"table": None}}}
