"""Guards and option paths of the recruitment / replication panel and filter
helpers in spacr.submodules."""
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import submodules as sm  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _objects(n_per_well=4):
    """Object rows in four wells: two conditions, two replicates each."""
    rows = []
    for cond, rep, col, base in (("a", "r1", "c1", 1.0), ("a", "r2", "c2", 1.2),
                                 ("b", "r1", "c3", 2.0), ("b", "r2", "c4", 2.3)):
        for i in range(n_per_well):
            rows.append(dict(plateID="p1", rowID="r1", columnID=col, condition=cond,
                             rep=rep, value=base + 0.1 * i, kind=f"k{i % 2}"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- panel settings
def test_panel_where_names_a_missing_column():
    with pytest.raises(KeyError, match="panel 'where' column 'nope' is not in the table"):
        sm._panel_where(_objects(), {"nope": 1})


def test_check_panels_setting_accepts_one_dict_and_rejects_other_entries():
    out = sm._check_panels_setting({"value": "x"})
    assert out == [{"value": "x", "name": "panel_1"}]
    with pytest.raises(TypeError, match=r"panels\[1\] must be a dict, got str"):
        sm._check_panels_setting([{"name": "ok"}, "bad"])


def test_unit_frame_checks_the_replicate_column():
    df = _objects()
    with pytest.raises(KeyError, match="replicate_column 'missing' is not a column"):
        sm._unit_frame(df, "value", "condition", "missing")
    with pytest.raises(ValueError, match="cannot also be the panel group column"):
        sm._unit_frame(df, "value", "condition", "condition")
    unit = sm._unit_frame(df, "value", "condition", "rep")
    assert len(unit) == 4 and set(unit["n_wells"]) == {1}


def test_bar_panel_from_objects_names_a_missing_value_column():
    with pytest.raises(KeyError, match="panel 'p': value column 'absent' is not in the table"):
        sm._bar_panel_from_objects(_objects(), {"name": "p"}, {}, "absent", "condition")


# --------------------------------------------------------------------------- writing panels
def test_write_panels_takes_list_figures_skips_empty_ones_and_can_be_quiet(tmp_path, capsys):
    spec = sm._bar_panel_from_objects(_objects(), {"name": "p"}, {}, "value", "condition")
    out = sm._write_panels_and_figures(
        {"P01": spec}, {"P01": "values"}, str(tmp_path / "panels"), str(tmp_path / "figs"),
        {"F_list": ["P01"], "F_empty": {"panels": ["P09"], "skip_missing": True}},
        verbose=False)
    assert "Wrote 1 panel(s)" not in capsys.readouterr().out
    assert sorted(out["figures"]) == ["F_list"]
    assert out["figures"]["F_list"] == [str(tmp_path / "figs" / "F_list.pdf"),
                                        str(tmp_path / "figs" / "F_list.png")]
    assert os.path.isdir(out["panels"]["P01"])


# --------------------------------------------------------------------------- filter log
def test_filter_log_without_steps_and_without_the_group_column():
    df = _objects().drop(columns="condition")
    log = sm._FilterLog(df, unit="objects")
    frame = log.frame()
    assert frame.to_dict("records") == [{"step": "spaCR output", "objects_remaining": 16,
                                         "objects_removed": 0}]
    assert log.kept_per_condition(df).empty


def test_edge_cut_mask_names_the_missing_columns():
    with pytest.raises(KeyError, match="exclude_edge_objects needs the columns"):
        sm._edge_cut_mask(_objects(), "pathogen")


# --------------------------------------------------------------------------- confounder filter
def test_confounder_filter_rejects_unknown_columns_and_levels():
    log = sm._FilterLog(_objects())
    with pytest.raises(KeyError, match="confounder_filter column 'zz' is not in the table"):
        sm._apply_confounder_filter(_objects(), {"zz": [0, 1]}, log)
    with pytest.raises(ValueError, match="level must be 'object' or 'well'"):
        sm._apply_confounder_filter(_objects(), {"value": {"level": "plate"}}, log)


def test_confounder_filter_one_sided_quantiles_and_ranges(capsys):
    df = _objects().drop(columns="plateID")
    log = sm._FilterLog(df)
    # quantiles with only an upper bound: keep values <= the 50th percentile
    out = sm._apply_confounder_filter(df, {"value": {"quantiles": [None, 0.5]}}, log)
    assert out["value"].max() <= df["value"].median()
    # quantiles with only a lower bound
    out2 = sm._apply_confounder_filter(df, {"value": {"quantiles": [0.5, None]}}, log)
    assert out2["value"].min() >= df["value"].median()
    # a range with no lower bound, and one with a lower bound only
    out3 = sm._apply_confounder_filter(df, {"value": [None, 1.25]}, log)
    assert out3["value"].max() <= 1.25 and len(out3) == 4
    out4 = sm._apply_confounder_filter(df, {"value": [2.0]}, log)
    assert out4["value"].min() >= 2.0 and len(out4) == 8
    steps = log.frame()
    assert steps["note"].tolist()[-2:] == ["object-level range [None, 1.25]",
                                          "object-level range [2.0]"]


# --------------------------------------------------------------------------- recruitment filters
def test_recruitment_filters_log_field_qc_and_check_the_vacuole_count_column():
    df = _objects()
    before_qc = pd.concat([df, df.iloc[:3]], ignore_index=True)
    out, log = sm._recruitment_new_filters(
        df, {"confounder_filter": {"value": [None, 2.0]}}, "unused",
        field_qc_counts=(before_qc, df))
    steps = log.frame()
    assert steps["step"].tolist() == ["spaCR output (after object filters)", "field_qc",
                                      "confounder_filter value"]
    assert steps["objects_removed"].tolist()[1] == 3
    assert (out["value"] <= 2.0).all()
    with pytest.raises(KeyError, match="needs the 'pathogen_prcfo_count' column"):
        sm._recruitment_new_filters(df, {"max_vacuoles_per_cell": 2}, "unused")
    # panels alone switch the log on without any filter step
    same, log = sm._recruitment_new_filters(df, {"panels": [{"name": "p"}]}, "unused")
    assert len(same) == len(df) and len(log.frame()) == 1


# --------------------------------------------------------------------------- control panels
def _control_frame():
    df = _objects()
    df["cell_channel_0_mean_intensity"] = 100.0 + np.arange(len(df))
    df["pathogen_channel_0_cytoplasm_mean_ratio"] = 1.0 + 0.05 * np.arange(len(df))
    df["host_cells"] = "hela"
    df["treatment"] = df["condition"]
    df["pathogen"] = "tg"
    return df


def test_control_palette_colours_conditions_by_a_metadata_column():
    df = _control_frame()
    df["drug"] = np.where(df["condition"] == "a", "dmso", "x")
    palette = sm._control_palette(df, "condition", ["a", "b"], "drug")
    assert palette["a"] != palette["b"]
    assert sm._control_palette(df, "condition", ["a", "b"], "absent") is None
    assert sm._control_palette(df, "condition", ["a", "b"], None) is None


def test_control_specs_with_replicates_inside_each_condition():
    df = _control_frame()
    specs, names, figures = sm._recruitment_control_specs(
        df, {"channel_dims": [0], "channel_of_interest": 0, "replicate_column": "rep",
             "control_color_by": "treatment", "treatments": ["b", "a"]})
    # only the compartments and ratios present become panels
    assert sorted(names.values()) == ["channel_0_cell_mean_intensity", "objects_per_well",
                                      "pathogen_channel_0_cytoplasm_mean_ratio"]
    counts = specs[[k for k, v in names.items() if v == "objects_per_well"][0]]["data"]
    # two replicates per condition: objects per well are averaged per replicate
    assert set(counts.columns) >= {"condition", "rep", "objects_per_well", "n_wells"}
    assert len(counts) == 4 and (counts["objects_per_well"] == 4).all()
    # the plate map lists treatment b before a
    assert specs["C01"]["order"] == ["b", "a"]
    assert sorted(figures) == ["controls_channel_0_intensity", "controls_objects_per_well",
                               "controls_recruitment_ratios_channel_0"]


def test_recruitment_panels_without_controls_or_figures(tmp_path):
    df = _control_frame()
    out = sm._recruitment_panels(df, {"panels": [{"name": "v", "value": "value"}]}, str(tmp_path))
    assert sorted(out) == ["panels"]
    folder = out["panels"]["panels"]["P01"]
    assert os.path.basename(folder) == "P01_v"
    data = pd.read_csv(os.path.join(folder, "data_replicates.csv"))
    assert sorted(data["condition"].unique()) == ["a", "b"]
    with pytest.raises(KeyError, match=r"figures refer to unknown panel id\(s\) \['P07'\]"):
        sm._recruitment_panels(df, {"panels": [{"name": "v", "value": "value"}],
                                    "figures": {"F": ["P07"]}}, str(tmp_path))


# --------------------------------------------------------------------------- replication helpers
def _replication_db(path, n_fields=3):
    rows = []
    for f in range(n_fields):
        for i in range(5 + f):
            rows.append(dict(plateID="plate%5F1", rowID="r1", columnID="c1",
                             fieldID=str(f), object_label=len(rows) + 1,
                             pathogen_area=100.0 + len(rows)))
    with sqlite3.connect(path) as con:
        pd.DataFrame(rows).to_sql("pathogen", con, index=False)
    return pd.DataFrame(rows)


def test_replication_filters_field_qc_with_a_plate_override_and_area_guard(tmp_path):
    db = str(tmp_path / "m.db")
    frame = _replication_db(db).assign(plateID="plate1")
    out, log, flags = sm._replication_filters(
        frame, {"field_qc": True, "field_qc_min_objects": 6}, "pathogen_area",
        lambda f: np.zeros(len(f), bool), [(db, "pathogen", "plate1")],
        unit="objects", first_step="start")
    assert set(flags["plateID"]) == {"plate1"}
    assert flags.set_index("fieldID")["field_qc_reason"].to_dict()["0"] == "n_objects_field<6"
    assert "n_objects_field" in out.columns and (out["fieldID"] != "0").all()
    assert log.frame()["step"].tolist() == ["start", "field_qc"]
    with pytest.raises(KeyError, match="max_area_quantile needs the column 'missing_area'"):
        sm._replication_filters(frame, {"max_area_quantile": 0.9}, "missing_area",
                                None, [], unit="objects", first_step="start")


def test_volume_bin_label_that_is_not_a_range_is_kept():
    assert sm._volume_bin_as_area("non_power_of_two") == "non_power_of_two"
    assert sm._volume_bin_as_area(">abc") == ">abc"


def test_replication_panels_check_group_replicate_and_figure_ids(tmp_path):
    table = _objects().assign(bin=lambda d: d["kind"])
    with pytest.raises(KeyError, match="group column 'nope' is not in the table"):
        sm._replication_panels(table, {"panels": [{"name": "p", "group": "nope"}]}, "bin",
                               ["k0", "k1"], "value", "value", "", str(tmp_path))
    with pytest.raises(KeyError, match="replicate_column 'gone' is not a column of the table"):
        sm._replication_panels(table, {"panels": [{"name": "p"}], "replicate_column": "gone"},
                               "bin", ["k0", "k1"], "value", "value", "", str(tmp_path))
    with pytest.raises(KeyError, match=r"figures refer to unknown panel id\(s\) \['P04'\]"):
        sm._replication_panels(table, {"panels": [{"name": "p", "n_resamples": 50}],
                                       "figures": {"F": {"panels": ["P04"]}}},
                               "bin", ["k0", "k1"], "value", "value", "", str(tmp_path))


# --------------------------------------------------------------------------- analyze_replication
def test_direct_count_replication_needs_the_area_for_the_quantile_filter(tmp_path):
    from tests.test_replication_assay import settings_for, write_db

    src = write_db(tmp_path / "plate1", [{"row": "r1", "column": c, "cell": 1, "n": 2}
                                         for c in ("c1", "c2")])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select * from pathogen", con).drop(columns="pathogen_area")
        p.to_sql("pathogen", con, index=False, if_exists="replace")
    with pytest.raises(KeyError, match="max_area_quantile needs 'pathogen_area' in 'pathogen'"):
        sm.analyze_replication(settings_for(src, max_area_quantile=0.9))


def test_direct_count_replication_refuses_when_every_vacuole_is_filtered(tmp_path):
    from tests.test_replication_assay import settings_for, write_db

    src = write_db(tmp_path / "plate1", [{"row": "r1", "column": c, "cell": 1, "n": 2}
                                         for c in ("c1", "c2")])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select * from pathogen", con)
        p["pathogen_distance_to_field_edge"] = 0.0
        p.to_sql("pathogen", con, index=False, if_exists="replace")
    with pytest.raises(ValueError, match="No vacuoles left after field_qc"):
        sm.analyze_replication(settings_for(src, exclude_edge_objects=True))


def test_size_proxy_replication_refuses_when_every_object_is_filtered(tmp_path):
    from tests.test_cov_submodules_endodyogeny import (
        _write_endodyogeny_db, AREAS_C1, AREAS_C2, _settings)

    root = tmp_path / "plate1"
    db = _write_endodyogeny_db(root, {"c1": AREAS_C1, "c2": AREAS_C2})
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select * from pathogen", con)
        p["pathogen_equivalent_diameter_area"] = 20.0
        p["pathogen_distance_to_field_edge"] = 1.0
        p.to_sql("pathogen", con, index=False, if_exists="replace")
    with pytest.raises(ValueError, match="No objects left after field_qc"):
        sm.analyze_replication(_settings(str(root), replication_method="size_proxy",
                                         exclude_edge_objects=True))


# --------------------------------------------------------------------------- analyze_recruitment
def test_recruitment_field_columns_as_covariates_without_field_qc(tmp_path):
    from tests.test_cov_submodules_recruitment_plaques import (
        _merged_recruitment_df, _base_settings, _Recorder)
    import spacr.io as SIO
    import spacr.plot as SPLT

    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    df = _merged_recruitment_df(n_cells=6)
    for chan in range(4):
        df[f"cytoplasm_channel_{chan}_mean_intensity"] += np.arange(len(df)) % 5
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        con.execute("create table cell (plateID, rowID, columnID, fieldID, cell_id)")
        con.execute("create table pathogen (plateID, rowID, columnID, fieldID, cell_id)")
        for rid, cid, n in (("r1", "c1", 20), ("r4", "c4", 30)):
            for i in range(n):
                con.execute("insert into cell values (?,?,?,?,?)", ("plate1", rid, cid, "1", i + 1))
                con.execute("insert into pathogen values (?,?,?,?,?)", ("plate1", rid, cid, "1", i + 1))
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_controls", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        cells, _ = sm.analyze_recruitment(_base_settings(
            src, field_qc=False, confounder_covariates=["n_cells_field"]))
    finally:
        mp.undo()
    res = src / "results"
    # the field counts are attached for the screen, but no field is dropped
    assert len(cells) == len(df)
    assert set(cells["n_cells_field"]) == {20, 30}
    assert not (res / "field_qc.csv").exists()
