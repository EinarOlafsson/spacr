"""Live/dead viability on the inputs at its edges: signals too small or too
flat to split, plates too small to fit, a run measured on cells rather
than nuclei, time courses, plate maps that name their plate, and control
scaling that cannot be done. Each ends in a named refusal or a table that
says what it could and could not decide."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m
from spacr.tabular import write_database, write_table
from tests.test_live_dead_viability import _signal_table


def test_signals_too_small_or_too_flat_are_not_split():
    assert np.isnan(m._two_population_fit(np.arange(5.0))["cut"])
    lonely = np.array([0.0] * 20 + [100.0])
    assert m._two_population_fit(lonely)["bimodal"] is False
    assert m._stain_scale([]) == 1.0
    flat = m._stain_cut(np.full(30, 4.0), single_is_positive=True)
    assert flat.source == "none" and np.isnan(flat.threshold)
    one_positive = m._stain_cut(np.array([-3.0, -2.0, 0.0, 5.0]),
                                single_is_positive=True, log_scale=True)
    assert one_positive.source == "none"


def test_signals_and_morphology_need_their_columns():
    table = _signal_table(n_live=30, n_dead=5)
    with pytest.raises(ValueError, match="no nucleus_channel_7_mean"):
        m._object_signal(table, "nucleus", 7)
    bare = table.drop(columns="nucleus_channel_1_outside_percentile_50")
    assert np.allclose(m._object_signal(bare, "nucleus", 1),
                       bare["nucleus_channel_1_mean_intensity"])
    with pytest.raises(ValueError, match="no nucleus_area"):
        m._condensation_score(table.assign(nucleus_area=np.nan), 0)
    dark = table.assign(nucleus_channel_0_mean_intensity=0.0,
                        nucleus_channel_0_outside_percentile_50=0.0)
    assert m._condensation_score(dark, 0).isna().all()
    assert m._viability_manual("[5, 7]") == (5.0, 7.0)


def test_small_plates_borrow_one_pooled_cut():
    big = _signal_table(n_live=200, n_dead=40, plates=("big",))
    small = _signal_table(n_live=4, n_dead=1, plates=("s1", "s2"), seed=3)
    objects = pd.concat([big, small], ignore_index=True)
    signal = m._object_signal(objects, "nucleus", 1)
    _positive, thresholds, cuts = m._split_by_plate(
        objects, signal, single_is_positive=False)
    assert cuts[("s1",)].source == "pooled"
    assert cuts[("s1",)].threshold == cuts[("s2",)].threshold
    assert np.isfinite(thresholds).all()


def test_time_courses_are_keyed_per_time_point():
    table = pd.DataFrame({"plateID": ["p", "p"], "rowID": ["r1", "r1"],
                          "columnID": ["c1", "c1"], "timeID": [1, 2]})
    assert m._viability_well_keys(table)[-1] == "timeID"
    assert m._plate_key(table).tolist() == ["p_1", "p_2"]


def test_the_measured_fields_skip_what_is_not_a_field(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    np.save(tmp_path / "junk.npy", np.zeros(2))
    np.save(tmp_path / "plate1_A01_1.npy", np.zeros(2))
    fields = m._measured_fields({"src": str(tmp_path)})
    assert fields[["plateID", "rowID", "columnID"]].values.tolist() == [
        ["plate1", "r1", "c1"]]


def test_control_scaling_that_fails_falls_back_to_the_negative_control(
        monkeypatch):
    import spacr.qt.widgets.dose_response as dr

    def refuse(*_a, **_k):
        raise dr.DoseResponseError("controls overlap")

    monkeypatch.setattr(dr, "normalise_to_controls", refuse)
    wells = pd.DataFrame({"plate_key": ["p"] * 3,
                          "role": ["negative", "positive", "sample"],
                          "n_live": [10, 2, 5], "viability": [1.0, 0.2, 0.5]})
    out = m._cytotoxicity_index(wells)
    assert set(out["cytotoxicity_basis"]) == {"negative control"}
    assert out["cytotoxicity_index"].round(6).tolist() == [0.0, 80.0, 50.0]


def test_a_field_list_without_a_time_column_still_fills_the_wells():
    table = pd.DataFrame({
        "plateID": ["p"], "rowID": ["r1"], "columnID": ["c1"],
        "timeID": [1], "viability_state": ["live"], "infected": [np.nan]})
    fields = pd.DataFrame({"plateID": ["p"], "rowID": ["r2"],
                           "columnID": ["c2"]})
    wells = m._viability_by_well(table, {}, fields)
    assert len(wells) == 2
    assert wells.set_index("rowID").loc["r2", "n_objects"] == 0


def test_qc_skips_a_stain_that_has_no_cut_for_the_plate():
    wells = pd.DataFrame({"plateID": ["p"], "plate_key": ["p_1"],
                          "timeID": ["1"], "role": ["sample"],
                          "viability": [0.5], "live_cell_index": [1.0],
                          "cytotoxicity_index": [0.0], "n_live": [3]})
    cut = m._PopulationCut(5.0, "mixture", 3.0, 0.5, 10)
    qc = m._viability_qc(wells, {"dead": {("other",): cut}})
    assert "dead_threshold" not in qc.columns
    assert qc.iloc[0]["timeID"] == "1"
    assert m._lookup_cut({("p", "2"): cut}, {"plateID": "p",
                                             "timeID": "1"}) is None


def test_a_plate_map_with_its_own_plate_column_is_joined_per_plate(tmp_path):
    wells = pd.DataFrame({"plateID": ["p1", "p2"], "rowID": ["r1", "r1"],
                          "columnID": ["c1", "c1"]})
    path = tmp_path / "map.csv"
    write_table(pd.DataFrame({"plateID": ["p1", "p2"],
                              "wellID": ["A01", "A01"],
                              "compound": ["drug", "vehicle"],
                              "concentration": [1.0, 0.0]}), str(path))
    joined = m._read_plate_map(str(path), wells)
    assert joined["compound"].tolist() == ["drug", "vehicle"]


def test_dose_response_of_nothing_measured_is_empty():
    empty = pd.DataFrame({"compound": pd.Series([], dtype=str),
                          "concentration": pd.Series([], dtype=float),
                          "viability": pd.Series([], dtype=float),
                          "cytotoxicity_index": pd.Series([], dtype=float),
                          "infection_live": pd.Series([], dtype=float)})
    curves, selectivity, fits = m._viability_dose_response(empty)
    assert curves.empty and selectivity.empty and fits == {}
    assert m._viability_dose_figure({}) is None


def _db(tmp_path, frame, table):
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    write_database(frame, str(db), table, if_exists="replace")
    return str(db)


def test_a_run_measured_on_cells_is_called_by_its_stain_only(tmp_path):
    nuclei = _signal_table(n_live=120, n_dead=30)
    cells = nuclei.rename(columns=lambda c: c.replace("nucleus_", "cell_"))
    cells = cells.drop(columns="prcf")
    db = _db(tmp_path, cells, "cell")
    with pytest.raises(ValueError, match="needs measured nuclei"):
        m._classify_viability(db, {"channels": [0, 1, 2]}, plot=False)
    table, report = m._classify_viability(
        db, {"channels": [0, 1, 2], "viability_dead_channel": 1,
             "src": str(tmp_path / "merged")}, plot=False)
    assert report["object_type"] == "cell"
    assert set(table["object_type"]) == {"cell"}
    called = table["viability_state"] == "dead"
    truth = cells["state"] == "dead"
    assert (called == truth).mean() >= 0.95


def test_morphology_reads_the_first_channel_and_draws_its_log_cut(tmp_path):
    nuclei = _signal_table(n_live=200, n_dead=40)
    nuclei["timeID"] = 1
    db = _db(tmp_path, nuclei, "nucleus")
    table, report = m._classify_viability(
        db, {"channels": [0, 1, 2], "nucleus_channel": ""}, plot=True)
    assert report["method"] == "morphology"
    assert set(table["viability_state"]) <= {"live", "dead"}
    figures = {name.split("/")[-1].split(".")[0]
               for name in report["figures"]}
    assert any(name.startswith("viability_thresholds_p1") for name in figures)
    with sqlite3.connect(db) as conn:
        wells = pd.read_sql_query(f"SELECT * FROM {m._VIABILITY_WELL_TABLE}",
                                  conn)
    assert set(wells["timeID"].astype(str)) == {"1"}


def _dosed_wells():
    rows = []
    for compound in ("A", "B"):
        for dose in (0.0, 0.1, 1.0, 10.0, 100.0):
            for rep in range(2):
                live = 1.0 / (1.0 + dose / (3.0 if compound == "A" else 30.0))
                rows.append({"compound": compound, "concentration": dose,
                             "viability": live,
                             "cytotoxicity_index": 100.0 * (1 - live),
                             "infection_live": (0.8 / (1.0 + dose)
                                                if compound == "A"
                                                else np.nan)})
    return pd.DataFrame(rows)


def test_selectivity_is_reported_and_a_missing_curve_is_drawn_refused():
    import matplotlib.pyplot as plt

    curves, selectivity, fits = m._viability_dose_response(_dosed_wells())
    assert set(fits) == {"viability", "cytotoxicity_index", "infection"}
    assert selectivity["compound"].tolist() == ["A", "B"]
    figure = m._viability_dose_figure(fits)
    try:
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert texts.count("refused") == 1, "B has no infection curve"
    finally:
        plt.close(figure)
    without = _dosed_wells().assign(cytotoxicity_index=np.nan)
    _curves, selectivity, fits = m._viability_dose_response(without)
    assert "cytotoxicity_index" not in fits and selectivity.empty


def test_qc_without_a_z_prime_still_lists_the_plate(monkeypatch):
    import spacr.qt.widgets.dose_response as dr

    def refuse(*_a, **_k):
        raise dr.DoseResponseError("no controls")

    monkeypatch.setattr(dr, "plate_reports", refuse)
    wells = pd.DataFrame({"plateID": ["p"], "plate_key": ["p"],
                          "role": ["sample"], "viability": [0.5],
                          "live_cell_index": [1.0],
                          "cytotoxicity_index": [0.0], "n_live": [3]})
    qc = m._viability_qc(wells, {})
    assert qc["plate_key"].tolist() == ["p"]
    assert qc["zprime_viability"].isna().all()


def test_a_whole_table_cut_is_drawn_without_picking_a_plate():
    import matplotlib.pyplot as plt

    table = pd.DataFrame({"dead_signal": np.r_[np.full(40, 5.0),
                                               np.full(10, 500.0)],
                          "viability_method": "stain"})
    cut = m._PopulationCut(100.0, "mixture", 4.0, 0.2, 50)
    figure = m._viability_threshold_figure(table, {"dead": {"all": cut}},
                                           "all", "whole run")
    try:
        assert figure.axes[0].get_title().startswith("dead: cut 100")
    finally:
        plt.close(figure)


def test_the_step_prints_the_controls_z_prime_per_plate(tmp_path, capsys):
    p1 = _signal_table(n_live=200, n_dead=40, plates=("p1",))
    dead = p1["state"] == "dead"
    p1["columnID"] = np.where(dead, np.where(np.arange(len(p1)) % 2, "c3",
                                             "c4"),
                              np.where(np.arange(len(p1)) % 2, "c1", "c2"))
    p1["prcf"] = "p1_r1_" + p1["columnID"] + "_f1"
    p2 = _signal_table(n_live=60, n_dead=10, plates=("p2",), seed=4)
    p2["columnID"] = "c1"
    p2["prcf"] = p2["plateID"] + "_r1_c1_f1"
    db = _db(tmp_path, pd.concat([p1, p2], ignore_index=True), "nucleus")
    settings = {"channels": [0, 1, 2], "viability_dead_channel": 1,
                "viability_negative_wells": ["c1", "c2"],
                "viability_positive_wells": ["c3", "c4"], "plot": False}
    table = m._run_viability_step(db, settings)
    printed = capsys.readouterr().out
    assert table is not None and "Viability (stain)" in printed
    assert "Viability controls, plate p1: Z'" in printed
    assert "plate p2" not in printed
    m._run_viability_step(db, {"channels": [0, 1, 2],
                               "viability_dead_channel": 1, "plot": False})
    assert "Viability controls" not in capsys.readouterr().out


def test_figures_skip_a_plate_map_that_draws_nothing(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import spacr.figures.plates as plates

    class _Panel:
        drawn = False

    monkeypatch.setattr(plates, "build_plates",
                        lambda *a, **k: (plt.figure(), _Panel()))
    db = _db(tmp_path, _signal_table(n_live=40, n_dead=10), "nucleus")
    table, report = m._classify_viability(
        db, {"channels": [0, 1, 2], "viability_dead_channel": 1}, plot=False)
    written = m._save_viability_figures(db, table, report["wells"],
                                        report["qc"], {}, {})
    assert [p.rsplit("/", 1)[-1].split(".")[0] for p in written] == [
        "viability_controls"]


def test_a_plate_map_by_row_and_column_is_read_as_it_is(tmp_path):
    wells = pd.DataFrame({"plateID": ["p1"], "rowID": ["r2"],
                          "columnID": ["c3"]})
    path = tmp_path / "map.csv"
    write_table(pd.DataFrame({"rowID": ["r2"], "columnID": ["c3"],
                              "compound": ["drug"],
                              "concentration": [2.0]}), str(path))
    assert m._read_plate_map(str(path), wells)["compound"].tolist() == [
        "drug"]


def test_fits_with_no_curve_draw_no_dose_figure(tmp_path):
    import types

    db = _db(tmp_path, _signal_table(n_live=40, n_dead=10), "nucleus")
    table, report = m._classify_viability(
        db, {"channels": [0, 1, 2], "viability_dead_channel": 1}, plot=False)
    written = m._save_viability_figures(
        db, table, report["wells"], report["qc"], {},
        {"viability": types.SimpleNamespace(fits=[])})
    assert not any("dose_response" in p for p in written)
    assert any("viability_controls" in p for p in written)


def test_objects_without_a_field_are_called_without_an_object_key(tmp_path):
    objects = _signal_table(n_live=80, n_dead=20).drop(
        columns=["prcf", "fieldID"])
    db = _db(tmp_path, objects, "nucleus")
    table, report = m._classify_viability(
        db, {"channels": [0, 1, 2], "viability_dead_channel": 1}, plot=False)
    assert "prcfo" not in table.columns
    assert (table["viability_state"] == "dead").sum() == pytest.approx(20,
                                                                       abs=2)
    assert len(report["wells"]) == 2
