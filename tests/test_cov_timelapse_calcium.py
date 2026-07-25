"""Coverage for the calcium-oscillation analysis block of ``spacr.timelapse``.

Covers ``plot_data``, ``infected_vs_noninfected``, ``summarize_per_well``,
``summarize_per_well_inf_non_inf`` and ``analyze_calcium_oscillations``
(lines ~856-1257 of ``spacr/timelapse.py``).

Everything runs off a tiny synthetic sqlite measurements DB built in-process:
10 timepoints, one plate, two "wells", five host cells that between them
exercise every branch of the analysis loop:

    obj 1  photobleaching baseline + two calcium spikes, infected (2 parasites)
    obj 2  fast monotonic decay      -> zero peaks (the ``len(peaks) == 0`` arm)
    obj 3  one spike, uninfected, second well
    obj 4  wildly fluctuating area   -> dropped by the size filter
    obj 5  only 5 of 10 timepoints   -> dropped by the transience filter

Two genuine defects, now fixed, are regression-tested here:

  * ``summarize_per_well`` used to graft ``cells_per_well`` onto the summary
    frame by *position* rather than by ``well_ID``, so any well whose peaks all
    have a null amplitude shifted every subsequent well's cell count.
  * ``analyze_calcium_oscillations`` used to read ``parasite_count``
    unconditionally even though that column only exists when ``pathogen=`` was
    supplied, so the documented default call raised ``KeyError``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_TIMEPOINTS = 10
PLATE, COLUMN, FIELD = "plate1", "c1", "f1"
MEASUREMENT = "cell_channel_1_mean_intensity"


@pytest.fixture(autouse=True)
def _close_figures():
    """No figure may survive a test (Agg still accumulates them)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Synthetic measurements DB
# ---------------------------------------------------------------------------

def _build_calcium_db(dirpath, with_pathogen=True, with_cytoplasm=True):
    """Write a `measurements.db` with cell/pathogen/cytoplasm tables.

    Returns the db path as a str.
    """
    t = np.arange(1, N_TIMEPOINTS + 1)
    # Global photobleaching envelope the module's exponential_decay must fit.
    base = 1000.0 * np.exp(-0.05 * t)

    rows = []

    def _add(row_id, obj_label, intensity, area, times=None):
        tt = t if times is None else times
        for i, ti in enumerate(tt):
            rows.append({
                "prcf": f"{PLATE}_{row_id}_{COLUMN}_{FIELD}_t{int(ti)}",
                "plateID": PLATE,
                "rowID": row_id,
                "column_name": COLUMN,
                "fieldID": FIELD,
                "timeid": int(ti),
                "object_label": obj_label,
                "cell_area": float(area[i]),
                MEASUREMENT: float(intensity[i]),
            })

    # obj 1 — two calcium spikes (t=4 and t=8), stable area, infected.
    i1 = base.copy()
    i1[3] *= 1.4
    i1[7] *= 1.4
    _add("r1", 1, i1, 500.0 + np.arange(N_TIMEPOINTS))

    # obj 2 — decays faster than the global fit, so every delta is negative
    #         and find_peaks returns nothing.
    _add("r1", 2, 900.0 * np.exp(-0.30 * t), np.full(N_TIMEPOINTS, 480.0))

    # obj 3 — one spike (t=6), different well, uninfected.
    i3 = 1100.0 * np.exp(-0.05 * t)
    i3[5] *= 1.5
    _add("r2", 3, i3, np.full(N_TIMEPOINTS, 520.0))

    # obj 4 — area swings by ~70% of the mean -> size filter drops it.
    area4 = np.array([100.0, 900.0, 150.0, 1200.0, 200.0,
                      1400.0, 90.0, 1000.0, 300.0, 1300.0])
    _add("r2", 4, base * 1.05, area4)

    # obj 5 — transient track, only 5 of the 10 timepoints.
    _add("r1", 5, base[:5] * 0.9, np.full(5, 510.0), times=t[:5])

    cell = pd.DataFrame(rows)

    db_path = os.path.join(str(dirpath), "measurements.db")
    con = sqlite3.connect(db_path)
    try:
        cell.to_sql("cell", con, index=False)

        if with_pathogen:
            prows = []
            for ti in t:
                for parasite in (1, 2):  # two parasites inside host cell 1
                    prows.append({
                        "plateID": PLATE, "rowID": "r1", "column_name": COLUMN,
                        "fieldID": FIELD, "timeid": int(ti),
                        "pathogen_cell_id": 1,
                        "object_label": parasite,
                        "pathogen_area": 30.0 + parasite,
                    })
            pd.DataFrame(prows).to_sql("pathogen", con, index=False)

        if with_cytoplasm:
            cyto = cell[["plateID", "rowID", "column_name", "fieldID",
                         "timeid", "object_label"]].copy()
            cyto["cytoplasm_area"] = 200.0
            cyto.to_sql("cytoplasm", con, index=False)
    finally:
        con.close()
    return db_path


def _peak_details_frame():
    """Hand-built peak-details frame with one null-amplitude (peak-less) cell.

    Wells:
        A_01 -> cells 1 and 2; cell 1 has two peaks, cell 2 has none
        B_02 -> cell 3 with one peak
    """
    return pd.DataFrame({
        "ID": ["p1_A_01_f1_1", "p1_A_01_f1_1", "p1_A_01_f1_2", "p1_B_02_f1_3"],
        "time": [4.0, 8.0, np.nan, 6.0],
        "amplitude": [0.5, 0.7, np.nan, 0.3],
        "delta": [0.5, 0.7, np.nan, 0.3],
        "AUC": [1.0, 1.0, 2.0, 3.0],
        "AUC_positive": [1.5, 1.5, 2.5, 3.5],
        "AUC_peak": [0.25, 0.35, np.nan, 0.15],
        "infected": [2.0, 2.0, 0.0, 0.0],
    })


# ---------------------------------------------------------------------------
# plot_data
# ---------------------------------------------------------------------------

def test_plot_data_draws_delta_column_against_time():
    from spacr.timelapse import plot_data

    group = pd.DataFrame({
        "time": [1, 2, 3, 4],
        "delta_" + MEASUREMENT: [0.0, 0.2, -0.1, 0.4],
    })
    fig, ax = plt.subplots()
    plot_data(MEASUREMENT, group, ax, "Infected", marker="x", linestyle="--")

    assert len(ax.lines) == 1
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), [1, 2, 3, 4])
    np.testing.assert_allclose(line.get_ydata(), [0.0, 0.2, -0.1, 0.4])
    assert line.get_label() == "Infected"
    assert line.get_marker() == "x"
    assert line.get_linestyle() == "--"


def test_plot_data_defaults_are_circle_marker_solid_line():
    from spacr.timelapse import plot_data

    group = pd.DataFrame({"time": [0, 1], "delta_x": [1.0, 2.0]})
    fig, ax = plt.subplots()
    plot_data("x", group, ax, "Uninfected")

    assert ax.lines[0].get_marker() == "o"
    assert ax.lines[0].get_linestyle() == "-"


def test_plot_data_raises_when_delta_column_absent():
    from spacr.timelapse import plot_data

    fig, ax = plt.subplots()
    with pytest.raises(KeyError):
        plot_data("missing", pd.DataFrame({"time": [0, 1]}), ax, "lbl")


# ---------------------------------------------------------------------------
# infected_vs_noninfected
# ---------------------------------------------------------------------------

def _result_frame_for_split():
    """Three tracks: two ever-infected, one never infected."""
    recs = []
    for obj, counts in (
        ("p1_A_01_f1_1", [0, 1, 2, 2]),   # infected at some point
        ("p1_A_01_f1_2", [3, 3, 3, 3]),   # infected throughout
        ("p1_B_02_f1_3", [0, 0, 0, 0]),   # never infected
    ):
        for i, (ti, cnt) in enumerate(zip([1, 2, 3, 4], counts)):
            recs.append({
                "plate_row_column_field_object": obj,
                "time": ti,
                "parasite_count": cnt,
                "delta_" + MEASUREMENT: 0.1 * i,
            })
    return pd.DataFrame(recs)


def test_infected_vs_noninfected_splits_tracks_across_two_axes():
    from spacr.timelapse import infected_vs_noninfected

    df = _result_frame_for_split()
    infected_vs_noninfected(df, MEASUREMENT)

    fig = plt.gcf()
    axs = fig.axes
    assert len(axs) == 2
    # Two ever-infected tracks on the top axis, one never-infected below.
    assert len(axs[0].lines) == 2
    assert len(axs[1].lines) == 1
    assert axs[0].get_title() == "Cells Infected at Some Time"
    assert axs[1].get_title() == "Cells Never Infected"
    for ax in axs:
        assert ax.get_ylabel() == "Normalized Delta " + MEASUREMENT
        assert list(ax.get_xticks()) == [1, 2, 3, 4]
    # Infected series use the 'x' marker, uninfected the default 'o'.
    assert axs[0].lines[0].get_marker() == "x"
    assert axs[1].lines[0].get_marker() == "o"
    np.testing.assert_allclose(axs[1].lines[0].get_ydata(),
                               [0.0, 0.1, 0.2, 0.30000000000000004])


def test_infected_vs_noninfected_all_uninfected_leaves_top_axis_empty():
    from spacr.timelapse import infected_vs_noninfected

    df = _result_frame_for_split()
    df["parasite_count"] = 0
    infected_vs_noninfected(df, MEASUREMENT)

    axs = plt.gcf().axes
    assert len(axs[0].lines) == 0
    assert len(axs[1].lines) == 3


# ---------------------------------------------------------------------------
# summarize_per_well
# ---------------------------------------------------------------------------

def test_summarize_per_well_counts_peaks_and_averages_numeric_columns():
    from spacr.timelapse import summarize_per_well

    out = summarize_per_well(_peak_details_frame())

    assert list(out["well_ID"]) == ["A_01", "B_02"]
    # Only rows with a non-null amplitude count as peaks.
    assert list(out["peaks_per_well"]) == [2, 1]
    # Both peaks in A_01 belong to the same track.
    assert list(out["unique_IDs_with_amplitude"]) == [1, 1]
    # cells_per_well is computed over the UNFILTERED frame, so the peak-less
    # cell 2 still counts towards well A_01.
    assert list(out["cells_per_well"]) == [2, 1]
    np.testing.assert_allclose(out["peaks_per_cell"], [1.0, 1.0])
    # Means are taken over the amplitude-bearing rows only.
    np.testing.assert_allclose(out["amplitude"], [0.6, 0.3])
    np.testing.assert_allclose(out["AUC"], [1.0, 3.0])
    np.testing.assert_allclose(out["time"], [6.0, 6.0])


def test_summarize_per_well_explodes_ID_into_identifier_columns():
    from spacr.timelapse import summarize_per_well

    df = _peak_details_frame()
    summarize_per_well(df)  # mutates its argument in place

    assert list(df["plateID"].unique()) == ["p1"]
    assert list(df["rowID"]) == ["A", "A", "A", "B"]
    assert list(df["columnID"]) == ["01", "01", "01", "02"]
    assert list(df["fieldID"].unique()) == ["f1"]
    assert list(df["object_number"]) == ["1", "1", "2", "3"]
    assert list(df["well_ID"]) == ["A_01", "A_01", "A_01", "B_02"]


def test_summarize_per_well_all_null_amplitudes_gives_empty_summary():
    from spacr.timelapse import summarize_per_well

    df = _peak_details_frame()
    df["amplitude"] = np.nan
    out = summarize_per_well(df)

    assert {"peaks_per_well", "cells_per_well", "peaks_per_cell"} <= set(out.columns)
    assert len(out) == 0


def test_summarize_per_well_cells_per_well_is_keyed_by_well():
    from spacr.timelapse import summarize_per_well

    # Well A_01 contributes a single peak-less cell, so it drops out of the
    # amplitude-filtered summary but stays in the cells-per-well groupby.
    df = pd.DataFrame({
        "ID": ["p1_A_01_f1_1",
               "p1_B_02_f1_1", "p1_B_02_f1_2", "p1_B_02_f1_3"],
        "amplitude": [np.nan, 0.5, 0.6, 0.7],
        "AUC": [1.0, 2.0, 3.0, 4.0],
        "infected": [0.0, 0.0, 1.0, 1.0],
    })
    out = summarize_per_well(df)

    assert list(out["well_ID"]) == ["B_02"]
    # B_02 really has three distinct cells, each contributing one peak.
    assert list(out["cells_per_well"]) == [3]
    np.testing.assert_allclose(out["peaks_per_cell"], [1.0])


# ---------------------------------------------------------------------------
# summarize_per_well_inf_non_inf
# ---------------------------------------------------------------------------

def test_summarize_per_well_inf_non_inf_splits_rows_by_infection_status():
    from spacr.timelapse import summarize_per_well_inf_non_inf

    out = summarize_per_well_inf_non_inf(_peak_details_frame())

    assert list(zip(out["well_ID"], out["infected_status"])) == [
        ("A_01", "infected"),
        ("A_01", "non_infected"),
        ("B_02", "non_infected"),
    ]
    assert list(out["peaks_per_well"]) == [2, 1, 1]
    assert list(out["cells_per_well"]) == [1, 1, 1]
    np.testing.assert_allclose(out["peaks_per_cell"], [2.0, 1.0, 1.0])
    np.testing.assert_allclose(out["amplitude"], [0.6, np.nan, 0.3])
    np.testing.assert_allclose(out["infected"], [2.0, 0.0, 0.0])


def test_summarize_per_well_inf_non_inf_treats_nan_infected_as_uninfected():
    from spacr.timelapse import summarize_per_well_inf_non_inf

    df = _peak_details_frame()
    df.loc[0, "infected"] = np.nan
    out = summarize_per_well_inf_non_inf(df)

    statuses = dict(zip(out["well_ID"] + "|" + out["infected_status"],
                        out["peaks_per_well"]))
    # Row 0 moved from 'infected' to 'non_infected'.
    assert statuses["A_01|infected"] == 1
    assert statuses["A_01|non_infected"] == 2


def test_summarize_per_well_inf_non_inf_all_infected_yields_one_row_per_well():
    from spacr.timelapse import summarize_per_well_inf_non_inf

    df = _peak_details_frame()
    df["infected"] = 4.0
    out = summarize_per_well_inf_non_inf(df)

    assert set(out["infected_status"]) == {"infected"}
    assert list(out["well_ID"]) == ["A_01", "B_02"]
    assert list(out["cells_per_well"]) == [2, 1]
    np.testing.assert_allclose(out["peaks_per_cell"], [1.5, 1.0])


# ---------------------------------------------------------------------------
# preprocess_pathogen_data
# ---------------------------------------------------------------------------

def test_preprocess_pathogen_data_counts_parasites_and_renames_cell_id():
    from spacr.timelapse import preprocess_pathogen_data

    df = pd.DataFrame({
        "plateID": [PLATE] * 5,
        "rowID": ["r1"] * 5,
        "column_name": [COLUMN] * 5,
        "fieldID": [FIELD] * 5,
        "timeid": [1, 1, 1, 1, 2],
        "pathogen_cell_id": [7, 7, 7, 9, 7],
        "object_label": [1, 2, 3, 1, 1],   # parasite ids, must be dropped
        "pathogen_area": [10.0, 20.0, 30.0, 5.0, 40.0],
        "note": ["a", "b", "c", "d", "e"],
    })
    out = preprocess_pathogen_data(df)

    # object_label now holds the HOST cell id, not the parasite id.
    assert sorted(out["object_label"]) == [7, 7, 9]
    assert "pathogen_cell_id" not in out.columns
    t1_host7 = out[(out["timeid"] == 1) & (out["object_label"] == 7)].iloc[0]
    assert t1_host7["parasite_count"] == 3
    # numeric columns are averaged, object columns take the first value
    assert t1_host7["pathogen_area"] == pytest.approx(20.0)
    assert t1_host7["note"] == "a"
    assert out[out["object_label"] == 9].iloc[0]["parasite_count"] == 1


# ---------------------------------------------------------------------------
# analyze_calcium_oscillations — happy path
# ---------------------------------------------------------------------------

def test_analyze_calcium_oscillations_full_run(tmp_path, capsys):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    out = analyze_calcium_oscillations(
        db, pathogen="pathogen", cytoplasm="cytoplasm", verbose=True)

    assert out is not None
    result_df, peak_details_df, fig = out

    # --- tracks that survived the transience + size filters --------------
    kept = set(result_df["plate_row_column_field_object"].unique())
    assert kept == {f"{PLATE}_r1_{COLUMN}_{FIELD}_1",
                    f"{PLATE}_r1_{COLUMN}_{FIELD}_2",
                    f"{PLATE}_r2_{COLUMN}_{FIELD}_3"}
    assert len(result_df) == 3 * N_TIMEPOINTS
    # delta is the first difference of the bleach-corrected trace
    assert "corrected_" + MEASUREMENT in result_df.columns
    assert "delta_" + MEASUREMENT in result_df.columns

    # --- peak table -------------------------------------------------------
    obj1 = peak_details_df[peak_details_df["ID"].str.endswith("_1")]
    assert list(obj1["time"]) == [4, 8]          # the two injected spikes
    assert (obj1["amplitude"] > 0.2).all()
    assert (obj1["infected"] == 2).all()         # two parasites in host cell 1

    obj2 = peak_details_df[peak_details_df["ID"].str.endswith("_2")]
    assert len(obj2) == 1                        # the no-peak placeholder row
    assert np.isnan(obj2["amplitude"].iloc[0])
    assert np.isnan(obj2["AUC_peak"].iloc[0])
    assert obj2["infected"].iloc[0] == 0
    assert np.isfinite(obj2["AUC"].iloc[0])
    # clipping negatives can only raise the AUC
    assert obj2["AUC_positive"].iloc[0] >= obj2["AUC"].iloc[0]

    obj3 = peak_details_df[peak_details_df["ID"].str.endswith("_3")]
    assert list(obj3["time"]) == [6]
    assert obj3["infected"].iloc[0] == 0

    # identifier columns exploded from prcf
    assert set(peak_details_df["plateID"]) == {PLATE}
    assert set(peak_details_df["rowID"]) == {"r1", "r2"}

    # --- emitted files ----------------------------------------------------
    results = tmp_path / "results"
    for name in ("peak_details", "results", "well_results",
                 "well_results_inf_non_inf"):
        assert (results / f"{name}.csv").is_file()
    assert (results / "figure_1.pdf").is_file()
    assert (results / "figure_2.pdf").is_file()   # pathogen branch

    well = pd.read_csv(results / "well_results.csv")
    assert sorted(well["well_ID"]) == ["r1_c1", "r2_c1"]

    # --- the returned figure ---------------------------------------------
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 3
    assert fig.axes[0].get_xlabel() == "Time"
    assert list(fig.axes[0].get_xticks()) == list(range(1, N_TIMEPOINTS + 1))

    printed = capsys.readouterr().out
    assert "After pathogen merge: 45 objects" in printed
    assert "After cytoplasm merge: 45 objects" in printed
    assert "Analyzing: 45 objects" in printed
    assert "removed group" in printed                # transience, verbose
    assert "Removed 1 objects due to size filter fluctuation" in printed
    assert "Removed 1 objects due to transience" in printed
    assert "Average number of peaks per infected cell: 2.00" in printed
    assert "Average number of peaks per non-infected cell: 1.00" in printed


def test_analyze_calcium_oscillations_without_optional_tables(tmp_path):
    """No cytoplasm table, no pathogen figure, quiet mode."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path, with_cytoplasm=False)
    result_df, peak_details_df, fig = analyze_calcium_oscillations(
        db, pathogen="pathogen", cytoplasm=None)

    assert "cytoplasm_area" not in result_df.columns
    assert (tmp_path / "results" / "figure_1.pdf").is_file()
    # figure_2 is only written on the pathogen-plot branch, which also ran
    assert (tmp_path / "results" / "figure_2.pdf").is_file()
    assert len(peak_details_df) == 4


def test_analyze_calcium_oscillations_num_lines_subsamples_the_plot(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    state = np.random.get_state()          # np.random.choice runs inside
    try:                                   # analyze -> seed it, then restore
        np.random.seed(0)
        result_df, _, fig = analyze_calcium_oscillations(
            db, pathogen="pathogen", num_lines=1)
    finally:
        np.random.set_state(state)

    # all three tracks are still analysed ...
    assert result_df["plate_row_column_field_object"].nunique() == 3
    # ... but only one is drawn on the summary axis
    assert len(fig.axes[0].lines) == 1


def test_analyze_calcium_oscillations_num_lines_above_track_count_plots_all(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    _, _, fig = analyze_calcium_oscillations(db, pathogen="pathogen",
                                             num_lines=99)
    assert len(fig.axes[0].lines) == 3


def test_analyze_calcium_oscillations_keeps_transient_tracks_when_disabled(tmp_path):
    """remove_transient=False lets the 5-timepoint track through."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    result_df, peak_details_df, _ = analyze_calcium_oscillations(
        db, pathogen="pathogen", remove_transient=False)

    kept = set(result_df["plate_row_column_field_object"].unique())
    assert f"{PLATE}_r1_{COLUMN}_{FIELD}_5" in kept
    assert len(kept) == 4
    assert set(peak_details_df["ID"]) == kept


def test_analyze_calcium_oscillations_loose_size_filter_keeps_wobbly_cell(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    result_df, _, _ = analyze_calcium_oscillations(
        db, pathogen="pathogen", fluctuation_threshold=5.0)

    assert f"{PLATE}_r2_{COLUMN}_{FIELD}_4" in set(
        result_df["plate_row_column_field_object"].unique())


def test_analyze_calcium_oscillations_high_peak_height_finds_no_peaks(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    _, peak_details_df, _ = analyze_calcium_oscillations(
        db, pathogen="pathogen", peak_height=10.0)

    # one placeholder row per surviving track, all amplitudes null
    assert len(peak_details_df) == 3
    assert peak_details_df["amplitude"].isna().all()
    assert peak_details_df["AUC_peak"].isna().all()
    assert list(peak_details_df["infected"]) == [1, 0, 0]


# ---------------------------------------------------------------------------
# analyze_calcium_oscillations — failure branches
# ---------------------------------------------------------------------------

def test_analyze_calcium_oscillations_curve_fit_failure_returns_none(tmp_path,
                                                                    monkeypatch,
                                                                    capsys):
    from spacr import timelapse as TL

    def _boom(*a, **k):
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(TL, "curve_fit", _boom)
    db = _build_calcium_db(tmp_path)

    assert TL.analyze_calcium_oscillations(db, pathogen="pathogen") is None
    assert "Curve fitting failed for the entire dataset" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_analyze_calcium_oscillations_no_suitable_cells_returns_none(tmp_path,
                                                                     capsys):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    # A negative fluctuation threshold can never be satisfied by std/mean.
    assert analyze_calcium_oscillations(
        db, pathogen="pathogen", fluctuation_threshold=-1.0, verbose=True) is None

    printed = capsys.readouterr().out
    assert "No suitable cells found for analysis" in printed
    assert "Removed 4 objects due to size filter fluctuation" in printed
    assert not (tmp_path / "results").exists()


def test_analyze_calcium_oscillations_everything_transient_returns_none(tmp_path,
                                                                        capsys):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    # threshold == total timepoints -> len(group) <= threshold for every track
    assert analyze_calcium_oscillations(
        db, pathogen="pathogen", transience_threshold=1.0) is None
    assert "No suitable cells found for analysis" in capsys.readouterr().out


def test_analyze_calcium_oscillations_missing_measurement_column(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path)
    with pytest.raises(KeyError):
        analyze_calcium_oscillations(db, measurement="not_a_column",
                                     pathogen="pathogen")


def test_analyze_calcium_oscillations_missing_db_raises(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    # sqlite3.connect happily creates an empty file; the cell table read fails.
    with pytest.raises(pd.errors.DatabaseError):
        analyze_calcium_oscillations(str(tmp_path / "nope.db"))


def test_analyze_calcium_oscillations_without_pathogen_table(tmp_path):
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_calcium_db(tmp_path, with_pathogen=False, with_cytoplasm=False)
    out = analyze_calcium_oscillations(db)

    assert out is not None
    result_df, peak_details_df, _ = out
    assert result_df["plate_row_column_field_object"].nunique() == 3
    assert (peak_details_df["infected"] == 0).all()
