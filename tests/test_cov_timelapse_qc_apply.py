"""CPU coverage for the timelapse motility back-end helpers:

    spacr.timelapse._apply_infection_intensity_qc
    spacr.timelapse._compute_velocities_and_well_summary
    spacr.timelapse._save_measurements_and_well_summary

The QC dispatcher is exercised with stand-in strategy helpers (so every
strategy / scope branch is reachable without dragging in xgboost or UMAP)
plus one end-to-end run through the real histogram strategy.

Everything here is CPU-only, offline and uses Agg-backed matplotlib.
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _qc_frame(n_cells=24, n_frames=2, plate="plate1", well="A01", field=1,
              pathogen_chan=2, low=120.0, high=1100.0, cell_offset=0):
    """Frame-level table with a clean bimodal pathogen-intensity split.

    The first half of the cells are labelled uninfected and are dim, the
    second half are labelled infected and are bright, so the histogram QC
    finds a threshold between the two populations.
    """
    rows = []
    for c in range(n_cells):
        infected = c >= n_cells // 2
        intensity = high if infected else low
        for f in range(n_frames):
            rows.append(
                {
                    "plateID": plate,
                    "wellID": well,
                    "fieldID": field,
                    "cellID": cell_offset + c + 1,
                    "frame": f + 1,
                    "infected": bool(infected),
                    f"cell_p95_intensity_ch{pathogen_chan}": float(intensity + c),
                }
            )
    return pd.DataFrame(rows)


def _make_fake_qc(name, calls, mode="passthrough"):
    """Build a stand-in for one of the _infection_qc_* strategy helpers."""

    def _qc(all_df, settings, infection_col, pathogen_chan, motility_dir):
        if "plateID" in all_df.columns and len(all_df):
            tag = str(all_df["plateID"].iloc[0])
        else:
            tag = name
        calls.append(
            {
                "name": name,
                "tag": tag,
                "n_rows": int(len(all_df)),
                "infection_col": infection_col,
                "pathogen_chan": pathogen_chan,
                "motility_dir": motility_dir,
                "settings": settings,
            }
        )
        settings["infection_hist_data"] = f"hist::{tag}"
        settings["infection_pca_data"] = f"pca::{tag}"
        settings["infection_xgb_importance"] = f"xgb::{tag}"
        settings["infection_intensity_qc_panel_type"] = name
        settings["infection_intensity_qc_panel_path"] = f"{name}::{tag}.png"

        df = all_df.copy()
        if mode == "nan_numeric":
            # first row NaN -> forces the fillna() branch, rest alternate 0/1
            vals = [float(i % 2) for i in range(len(df))]
            if vals:
                vals[0] = np.nan
            df["adjusted_infected"] = vals
            return df, "adjusted_infected"
        if mode == "strings":
            # non-castable -> astype(int) raises, astype(bool) fallback
            df["adjusted_infected"] = ["yes"] * len(df)
            return df, "adjusted_infected"
        if mode == "all_nan":
            df["adjusted_infected"] = np.nan
            return df, "adjusted_infected"
        return df, infection_col


    return _qc


def _patch_all_strategies(monkeypatch, calls, mode="passthrough"):
    import spacr.timelapse as tl
    for attr, label in (
        ("_infection_qc_histogram", "histogram"),
        ("_infection_qc_xgboost", "xgboost"),
        ("_infection_qc_pca_clustering", "pca"),
    ):
        monkeypatch.setattr(tl, attr, _make_fake_qc(label, calls, mode=mode))


_PAYLOAD_KEYS = (
    "infection_hist_data",
    "infection_pca_data",
    "infection_xgb_importance",
    "infection_intensity_qc_panel_type",
    "infection_intensity_qc_panel_path",
)


# ---------------------------------------------------------------------------
# _apply_infection_intensity_qc -- no-op paths
# ---------------------------------------------------------------------------

def test_qc_disabled_returns_input_untouched(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls)

    df = _qc_frame(n_cells=4, n_frames=1)
    settings = {
        "infection_intensity_qc": False,
        "infection_hist_data": "stale",
        "infection_pca_data": "stale",
        "infection_xgb_importance": "stale",
        "infection_intensity_qc_panel_type": "stale",
        "infection_intensity_qc_panel_path": "stale",
    }
    motility_dir = str(tmp_path / "never_created")

    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, motility_dir
    )

    assert out is df
    assert col == "infected"
    assert calls == []
    # every stale QC payload was reset
    assert all(settings[k] is None for k in _PAYLOAD_KEYS)
    # the output dir is only created once QC actually runs
    assert not os.path.exists(motility_dir)


def test_qc_without_pathogen_channel_is_noop(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls)

    df = _qc_frame(n_cells=4, n_frames=1)
    settings = {"infection_intensity_qc": True}
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", None, str(tmp_path / "mot")
    )
    assert out is df and col == "infected"
    assert calls == []


@pytest.mark.parametrize("scope", ["none", "off", "NONE", "Off"])
def test_qc_scope_none_skips_qc_but_resets_payloads(tmp_path, monkeypatch, scope):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls)

    df = _qc_frame(n_cells=4, n_frames=1)
    motility_dir = str(tmp_path / "mot")
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": scope,
        "infection_hist_data": "stale",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, motility_dir
    )

    assert out is df and col == "infected"
    assert calls == []
    assert settings["infection_hist_data"] is None
    # makedirs happens before the scope check
    assert os.path.isdir(motility_dir)


# ---------------------------------------------------------------------------
# strategy dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("hist", "histogram"),
        ("histogram", "histogram"),
        ("HISTAGRAM", "histogram"),
        ("xgboost", "xgboost"),
        ("XGB", "xgboost"),
        ("pca", "pca"),
        ("umap", "pca"),
        ("tsne", "pca"),
        ("banana", "histogram"),   # unknown -> histogram fallback
        (None, "histogram"),       # str(None) == 'none' -> also unknown
    ],
)
def test_qc_strategy_dispatch(tmp_path, monkeypatch, strategy, expected):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls)

    df = _qc_frame(n_cells=4, n_frames=1)
    motility_dir = str(tmp_path / "mot")
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": strategy,
        "infection_intensity_qc_scope": "combined",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 3, motility_dir
    )

    assert [c["name"] for c in calls] == [expected]
    assert calls[0]["pathogen_chan"] == 3
    assert calls[0]["motility_dir"] == motility_dir
    assert calls[0]["infection_col"] == "infected"
    # the helper is handed a COPY of settings, not the caller's dict
    assert calls[0]["settings"] is not settings
    assert col == "infected"          # passthrough mode -> no adjusted column
    assert "adjusted_infected" not in out.columns
    assert len(out) == len(df)
    assert os.path.isdir(motility_dir)
    # payloads written by the helper are propagated back onto the real settings
    assert settings["infection_intensity_qc_panel_type"] == expected
    assert settings["infection_hist_data"] == "hist::plate1"


def test_qc_default_strategy_is_histogram(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls)
    df = _qc_frame(n_cells=3, n_frames=1)
    settings = {"infection_intensity_qc": True}
    _apply_infection_intensity_qc(df, settings, "infected", 1, str(tmp_path / "m"))
    assert [c["name"] for c in calls] == ["histogram"]


# ---------------------------------------------------------------------------
# the three "single global run" code paths:
#   * scope == combined
#   * unknown scope -> combined behaviour
#   * plate/well scope with the grouping columns missing -> combined behaviour
# ---------------------------------------------------------------------------

def _single_run_case(case, tmp_path):
    """Return (df, settings) for one of the three global-run entry points."""
    df = _qc_frame(n_cells=4, n_frames=1)
    settings = {"infection_intensity_qc": True, "infection_intensity_strategy": "hist"}
    if case == "combined":
        settings["infection_intensity_qc_scope"] = "combined"
    elif case == "global":
        settings["infection_intensity_qc_scope"] = "GLOBAL"
    elif case == "all":
        settings["infection_intensity_qc_scope"] = "all"
    elif case == "unknown":
        settings["infection_intensity_qc_scope"] = "sideways"
    elif case == "missing_group_cols":
        settings["infection_intensity_qc_scope"] = "plate"
        df = df.drop(columns=["plateID"])
    elif case == "missing_well_col":
        settings["infection_intensity_qc_scope"] = "per_well"
        df = df.drop(columns=["wellID"])
    else:  # pragma: no cover - guarded by the parametrisation
        raise AssertionError(case)
    return df, settings


@pytest.mark.parametrize(
    "case", ["combined", "global", "all", "unknown", "missing_group_cols",
             "missing_well_col"],
)
def test_global_run_fills_nan_and_casts_to_int(tmp_path, monkeypatch, case):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="nan_numeric")
    df, settings = _single_run_case(case, tmp_path)

    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )

    assert len(calls) == 1                     # one single global QC run
    assert calls[0]["n_rows"] == len(df)
    assert col == "adjusted_infected"
    assert not out["adjusted_infected"].isna().any()
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    # row 0 was NaN and its 'infected' label is False -> filled with 0;
    # the remaining rows keep the alternating 0/1 pattern.
    expected = [0] + [i % 2 for i in range(1, len(df))]
    assert out["adjusted_infected"].tolist() == expected
    assert settings["infection_xgb_importance"] == "xgb::" + calls[0]["tag"]


@pytest.mark.parametrize(
    "case", ["combined", "unknown", "missing_group_cols"],
)
def test_global_run_falls_back_to_bool_cast(tmp_path, monkeypatch, case):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="strings")
    df, settings = _single_run_case(case, tmp_path)

    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )

    assert len(calls) == 1
    assert col == "adjusted_infected"
    assert out["adjusted_infected"].dtype == bool
    assert out["adjusted_infected"].all()


def test_global_run_without_adjusted_column_keeps_helper_column(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="passthrough")
    df = _qc_frame(n_cells=4, n_frames=1)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": "combined",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert col == "infected"
    assert "adjusted_infected" not in out.columns
    assert out is not df                       # helper handed back a copy


def test_scope_none_value_defaults_to_combined(tmp_path, monkeypatch):
    """settings['infection_intensity_qc_scope'] = None -> 'combined'."""
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="passthrough")
    df = _qc_frame(n_cells=4, n_frames=1)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": None,
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert len(calls) == 1 and calls[0]["n_rows"] == len(df)
    assert col == "infected"


# ---------------------------------------------------------------------------
# grouped scopes
# ---------------------------------------------------------------------------

def _two_plate_frame():
    a = _qc_frame(n_cells=4, n_frames=1, plate="plateA", well="A01")
    b = _qc_frame(n_cells=4, n_frames=1, plate="plateB", well="B02", cell_offset=100)
    return pd.concat([a, b], ignore_index=True)


@pytest.mark.parametrize("scope", ["plate", "PER_PLATE", "plateid"])
def test_scope_plate_runs_qc_per_plate(tmp_path, monkeypatch, scope):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="nan_numeric")
    df = _two_plate_frame()
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": scope,
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )

    assert [c["tag"] for c in calls] == ["plateA", "plateB"]
    assert [c["n_rows"] for c in calls] == [4, 4]
    assert col == "adjusted_infected"
    assert len(out) == len(df)
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    assert not out["adjusted_infected"].isna().any()
    # payloads come from the FIRST processed group only
    assert settings["infection_hist_data"] == "hist::plateA"
    assert settings["infection_pca_data"] == "pca::plateA"
    assert settings["infection_intensity_qc_panel_path"] == "histogram::plateA.png"


@pytest.mark.parametrize("scope", ["well", "per_well"])
def test_scope_well_runs_qc_per_well(tmp_path, monkeypatch, scope):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="passthrough")
    df = pd.concat(
        [
            _qc_frame(n_cells=2, n_frames=1, plate="plateA", well="A01"),
            _qc_frame(n_cells=2, n_frames=1, plate="plateA", well="A02",
                      cell_offset=50),
            _qc_frame(n_cells=2, n_frames=1, plate="plateB", well="A01",
                      cell_offset=90),
        ],
        ignore_index=True,
    )
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": scope,
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )

    # one QC call per (plateID, wellID) pair
    assert len(calls) == 3
    assert [c["n_rows"] for c in calls] == [2, 2, 2]
    # no group produced adjusted labels -> the original column name survives
    assert col == "infected"
    assert len(out) == len(df)
    assert sorted(out["wellID"].unique()) == ["A01", "A02"]


def test_scope_group_skips_empty_categorical_groups(tmp_path, monkeypatch):
    """Unused categorical levels yield empty groups, which must be skipped."""
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="nan_numeric")
    df = _qc_frame(n_cells=4, n_frames=1, plate="plateA", well="A01")
    df["plateID"] = pd.Categorical(
        df["plateID"], categories=["plateA", "plateGHOST"]
    )
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": "plate",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, col = _apply_infection_intensity_qc(
            df, settings, "infected", 2, str(tmp_path / "mot")
        )

    # the empty 'plateGHOST' group never reached the QC helper
    assert [c["tag"] for c in calls] == ["plateA"]
    assert len(out) == 4
    assert col == "adjusted_infected"


def test_scope_group_with_empty_frame_returns_input(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="nan_numeric")
    df = _qc_frame(n_cells=2, n_frames=1).iloc[0:0]
    assert df.empty and "plateID" in df.columns
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": "well",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert calls == []
    assert out is df
    assert col == "infected"


def test_scope_group_all_nan_adjusted_keeps_original_column(tmp_path, monkeypatch):
    """adjusted_infected present but entirely NaN -> not treated as adjusted."""
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="all_nan")
    df = _two_plate_frame()
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": "plate",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert len(calls) == 2
    assert col == "infected"
    assert "adjusted_infected" in out.columns
    assert out["adjusted_infected"].isna().all()


def test_scope_group_bool_cast_fallback(tmp_path, monkeypatch):
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode="strings")
    df = _two_plate_frame()
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": "plate",
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert col == "adjusted_infected"
    assert out["adjusted_infected"].dtype == bool
    assert out["adjusted_infected"].all()
    assert len(out) == len(df)


# ---------------------------------------------------------------------------
# real histogram strategy, end to end through the dispatcher
# ---------------------------------------------------------------------------

def test_combined_scope_with_real_histogram_strategy(tmp_path):
    from spacr.timelapse import _apply_infection_intensity_qc

    df = _qc_frame(n_cells=24, n_frames=2, pathogen_chan=2)
    motility_dir = str(tmp_path / "motility")
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_qc_scope": "combined",
        "infection_intensity_qc_graphs": True,
    }

    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, motility_dir
    )

    assert col == "adjusted_infected"
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    # bright cells (second half) are called infected, dim ones are not
    bright = out["cellID"] > 12
    assert out.loc[bright, "adjusted_infected"].eq(1).all()
    assert out.loc[~bright, "adjusted_infected"].eq(0).all()

    payload = settings["infection_hist_data"]
    assert isinstance(payload, dict)
    assert payload["intensity_col"] == "cell_p95_intensity_ch2"
    # threshold must sit strictly between the dim (<=131) and bright (>=1112)
    # populations built by _qc_frame
    assert 131.0 < payload["thr_val"] <= 1112.0
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    png = settings["infection_intensity_qc_panel_path"]
    assert png is not None and os.path.isfile(png)
    assert os.path.dirname(png) == motility_dir


def test_plate_scope_with_real_histogram_strategy(tmp_path):
    from spacr.timelapse import _apply_infection_intensity_qc

    df = pd.concat(
        [
            _qc_frame(n_cells=24, n_frames=1, plate="plateA", well="A01"),
            _qc_frame(n_cells=24, n_frames=1, plate="plateB", well="B01",
                      cell_offset=100),
        ],
        ignore_index=True,
    )
    motility_dir = str(tmp_path / "motility")
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "hist",
        "infection_intensity_qc_scope": "plate",
        "infection_intensity_qc_graphs": False,   # keep it fast: no PNG
    }

    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, motility_dir
    )

    assert col == "adjusted_infected"
    assert len(out) == len(df)
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    # each plate was thresholded on its own, both recover the injected split
    for plate in ("plateA", "plateB"):
        sub = out[out["plateID"] == plate]
        assert sub["adjusted_infected"].sum() == 12
        assert (
            sub.loc[sub["infected"], "adjusted_infected"].eq(1).all()
        )
    # graphs disabled -> payload present but no PNG on disk
    assert settings["infection_intensity_qc_panel_path"] is None
    assert isinstance(settings["infection_hist_data"], dict)


# ---------------------------------------------------------------------------
# _compute_velocities_and_well_summary
# ---------------------------------------------------------------------------

def _track_rows(plate, well, field, cell, xs, ys, infected):
    return [
        {
            "plateID": plate,
            "wellID": well,
            "fieldID": field,
            "cellID": cell,
            "frame": i + 1,
            "cell_centroid-1": float(x),
            "cell_centroid-0": float(y),
            "infected": bool(infected),
        }
        for i, (x, y) in enumerate(zip(xs, ys))
    ]


def _tracks_frame(specs):
    rows = []
    for spec in specs:
        rows.extend(_track_rows(*spec))
    return pd.DataFrame(rows)


def test_velocity_missing_centroid_columns(tmp_path):
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _qc_frame(n_cells=2, n_frames=2)      # has no centroid columns
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", 1.0, 60.0
    )
    assert track_df.empty and well_df.empty
    assert per_well == {}
    assert unit == "px/frame"
    assert isinstance(track_df, pd.DataFrame)


def test_velocity_physical_units_and_straightness():
    from spacr.timelapse import _compute_velocities_and_well_summary

    # one perfectly straight track moving 3 px per frame
    df = _tracks_frame(
        [("plate1", "A01", 1, 1, [0, 3, 6, 9], [0, 0, 0, 0], True)]
    )
    # factor = (1/3 px per um) * (60 / 30 s) = 0.666...
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", pixels_per_um=3.0, seconds_per_frame=30.0
    )

    assert unit == "µm/min"
    assert len(track_df) == 1
    rec = track_df.iloc[0]
    assert rec["v_px_per_frame"] == pytest.approx(3.0)
    assert rec["velocity"] == pytest.approx(2.0)
    assert rec["straightness"] == pytest.approx(1.0)
    assert rec["infected"] is True or bool(rec["infected"]) is True
    assert track_df["velocity_unit"].unique().tolist() == ["µm/min"]

    assert list(per_well.keys()) == [("plate1", "A01")]
    tr = per_well[("plate1", "A01")][0]
    assert tr["x_px"].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert tr["y_px"].tolist() == [0.0, 0.0, 0.0, 0.0]

    assert len(well_df) == 1
    w = well_df.iloc[0]
    assert w["n_tracks"] == 1
    assert w["n_infected_tracks"] == 1
    assert w["n_uninfected_tracks"] == 0
    assert w["mean_velocity_all"] == pytest.approx(2.0)
    assert w["mean_velocity_infected"] == pytest.approx(2.0)
    assert np.isnan(w["mean_velocity_uninfected"])
    assert w["velocity_unit"] == "µm/min"


@pytest.mark.parametrize(
    "px_per_um,sec_per_frame", [(None, 30.0), (3.0, None), (None, None)]
)
def test_velocity_pixel_units_when_calibration_missing(px_per_um, sec_per_frame):
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [("plate1", "A01", 1, 1, [0, 4, 8], [0, 0, 0], False)]
    )
    track_df, _per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", px_per_um, sec_per_frame
    )
    assert unit == "px/frame"
    assert track_df.iloc[0]["velocity"] == pytest.approx(4.0)
    assert well_df.iloc[0]["n_uninfected_tracks"] == 1
    assert well_df.iloc[0]["n_infected_tracks"] == 0
    assert np.isnan(well_df.iloc[0]["mean_velocity_infected"])
    assert well_df.iloc[0]["mean_velocity_uninfected"] == pytest.approx(4.0)


def test_velocity_skips_short_and_nonfinite_tracks():
    from spacr.timelapse import _compute_velocities_and_well_summary

    nan = float("nan")
    df = _tracks_frame(
        [
            # single frame -> too short
            ("plate1", "A01", 1, 1, [0], [0], False),
            # all-NaN centroids -> no finite displacements
            ("plate1", "A01", 1, 2, [nan, nan, nan], [nan, nan, nan], False),
            # stationary -> zero path length -> NaN straightness
            ("plate1", "A01", 1, 3, [5, 5, 5], [7, 7, 7], False),
            # usable track
            ("plate1", "A01", 1, 4, [0, 2, 4], [0, 0, 0], True),
        ]
    )
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", None, None
    )

    assert sorted(track_df["cellID"].tolist()) == [3, 4]
    stationary = track_df.set_index("cellID").loc[3]
    assert stationary["v_px_per_frame"] == pytest.approx(0.0)
    assert np.isnan(stationary["straightness"])
    moving = track_df.set_index("cellID").loc[4]
    assert moving["straightness"] == pytest.approx(1.0)
    assert len(per_well[("plate1", "A01")]) == 2
    assert well_df.iloc[0]["n_tracks"] == 2
    assert well_df.iloc[0]["mean_velocity_all"] == pytest.approx(1.0)
    assert unit == "px/frame"


def test_velocity_no_usable_tracks_returns_empty():
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [
            ("plate1", "A01", 1, 1, [0], [0], False),
            ("plate1", "A01", 1, 2, [3], [3], True),
        ]
    )
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", 2.0, 60.0
    )
    assert track_df.empty and well_df.empty
    assert per_well == {}
    assert unit == "px/frame"          # never upgraded to physical units


def test_straightness_filter_drops_tracks_and_prunes_wells():
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [
            # A01: only a perfectly straight (artifact-like) track
            ("plate1", "A01", 1, 1, [0, 3, 6, 9], [0, 0, 0, 0], True),
            # A02: one straight track + one back-and-forth track
            ("plate1", "A02", 1, 2, [0, 3, 6, 9], [0, 0, 0, 0], True),
            ("plate1", "A02", 1, 3, [0, 5, 0, 5], [0, 0, 0, 0], False),
        ]
    )
    settings = {"straightness_filter": True, "straightness_threshold": 0.95}
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, settings, "infected", None, None
    )

    assert track_df["cellID"].tolist() == [3]
    assert track_df.iloc[0]["straightness"] == pytest.approx(5.0 / 15.0)
    # A01 lost every track and was removed entirely; A02 kept the wiggly one
    assert list(per_well.keys()) == [("plate1", "A02")]
    assert [t["cellID"] for t in per_well[("plate1", "A02")]] == [3]
    assert len(well_df) == 1
    assert well_df.iloc[0]["wellID"] == "A02"
    assert well_df.iloc[0]["n_tracks"] == 1
    assert well_df.iloc[0]["n_infected_tracks"] == 0
    assert unit == "px/frame"


def test_straightness_filter_can_remove_every_track():
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [
            ("plate1", "A01", 1, 1, [0, 3, 6], [0, 0, 0], True),
            ("plate1", "A02", 1, 2, [0, 1, 2], [0, 0, 0], False),
        ]
    )
    settings = {"straightness_filter": True, "straightness_threshold": 0.5}
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, settings, "infected", 1.0, 60.0
    )
    assert track_df.empty
    assert per_well == {}
    assert well_df.empty
    # units were already resolved before the filter emptied the table
    assert unit == "µm/min"


def test_straightness_filter_off_keeps_straight_tracks():
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [("plate1", "A01", 1, 1, [0, 3, 6], [0, 0, 0], True)]
    )
    track_df, per_well, well_df, _unit = _compute_velocities_and_well_summary(
        df, {"straightness_threshold": 0.9}, "infected", None, None
    )
    assert len(track_df) == 1
    assert track_df.iloc[0]["straightness"] == pytest.approx(1.0)
    assert len(per_well[("plate1", "A01")]) == 1
    assert well_df.iloc[0]["n_tracks"] == 1


def test_well_summary_mixed_infection_means():
    from spacr.timelapse import _compute_velocities_and_well_summary

    df = _tracks_frame(
        [
            ("plate1", "A01", 1, 1, [0, 5, 0, 5], [0, 0, 0, 0], True),    # 5 px
            ("plate1", "A01", 2, 2, [0, 1, 0, 1], [0, 0, 0, 0], False),   # 1 px
        ]
    )
    track_df, per_well, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", None, None
    )
    assert len(track_df) == 2
    assert len(well_df) == 1
    w = well_df.iloc[0]
    assert w["n_tracks"] == 2
    assert w["n_infected_tracks"] == 1
    assert w["n_uninfected_tracks"] == 1
    assert w["mean_velocity_all"] == pytest.approx(3.0)
    assert w["mean_velocity_infected"] == pytest.approx(5.0)
    assert w["mean_velocity_uninfected"] == pytest.approx(1.0)
    assert unit == "px/frame"
    # both fields of the same well share one per_well entry
    assert len(per_well[("plate1", "A01")]) == 2


def test_infection_column_any_over_frames_marks_track_infected():
    from spacr.timelapse import _compute_velocities_and_well_summary

    # infected on a single frame only -> the whole track counts as infected
    rows = _track_rows("plate1", "A01", 1, 1, [0, 2, 4], [0, 0, 0], False)
    rows[1]["infected"] = True
    track_df, _pw, well_df, _u = _compute_velocities_and_well_summary(
        pd.DataFrame(rows), {}, "infected", None, None
    )
    assert bool(track_df.iloc[0]["infected"]) is True
    assert well_df.iloc[0]["n_infected_tracks"] == 1
    assert well_df.iloc[0]["n_uninfected_tracks"] == 0


# ---------------------------------------------------------------------------
# _save_measurements_and_well_summary
# ---------------------------------------------------------------------------

def test_save_measurements_with_well_summary(tmp_path):
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "proj"
    src.mkdir()
    all_df = pd.DataFrame(
        {"plateID": ["p1", "p1"], "wellID": ["A01", "A02"], "value": [1.5, 2.5]}
    )
    well_df = pd.DataFrame(
        {
            "plateID": ["p1"],
            "wellID": ["A01"],
            "n_tracks": [3],
            "velocity_unit": ["px/frame"],
        }
    )

    meas_dir, db_path = _save_measurements_and_well_summary(
        all_df, well_df, str(src), "motility"
    )

    assert meas_dir == os.path.join(str(src), "measurements")
    assert os.path.isdir(meas_dir)
    assert db_path == os.path.join(meas_dir, "measurements.db")
    assert os.path.isfile(db_path)

    con = sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"motility", "motility_well_motility"} <= tables
        got = pd.read_sql("SELECT * FROM motility", con)
        assert got["value"].tolist() == [1.5, 2.5]
        got_well = pd.read_sql("SELECT * FROM motility_well_motility", con)
        assert got_well["n_tracks"].tolist() == [3]
        assert got_well["velocity_unit"].tolist() == ["px/frame"]
    finally:
        con.close()


def test_save_measurements_without_well_summary(tmp_path):
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "proj"
    (src / "measurements").mkdir(parents=True)   # already-existing dir is fine
    all_df = pd.DataFrame({"a": [1, 2, 3]})

    meas_dir, db_path = _save_measurements_and_well_summary(
        all_df, pd.DataFrame(), str(src), "tracks"
    )

    assert os.path.isfile(db_path)
    con = sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "tracks" in tables
        assert "tracks_well_motility" not in tables
        assert pd.read_sql("SELECT * FROM tracks", con)["a"].tolist() == [1, 2, 3]
    finally:
        con.close()
    assert meas_dir.endswith("measurements")


def test_save_measurements_replaces_existing_table(tmp_path):
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "proj"
    src.mkdir()
    first = pd.DataFrame({"a": [1, 2, 3]})
    second = pd.DataFrame({"a": [9]})

    _save_measurements_and_well_summary(first, pd.DataFrame(), str(src), "tracks")
    _, db_path = _save_measurements_and_well_summary(
        second, pd.DataFrame(), str(src), "tracks"
    )

    con = sqlite3.connect(db_path)
    try:
        got = pd.read_sql("SELECT * FROM tracks", con)
    finally:
        con.close()
    assert got["a"].tolist() == [9]


def test_save_measurements_roundtrips_velocity_output(tmp_path):
    """The two helpers compose: velocities -> sqlite tables."""
    from spacr.timelapse import (
        _compute_velocities_and_well_summary,
        _save_measurements_and_well_summary,
    )

    df = _tracks_frame(
        [
            ("plate1", "A01", 1, 1, [0, 2, 4], [0, 0, 0], True),
            ("plate1", "A02", 1, 2, [0, 6, 12], [0, 0, 0], False),
        ]
    )
    _track_df, _pw, well_df, unit = _compute_velocities_and_well_summary(
        df, {}, "infected", None, None
    )
    src = tmp_path / "proj"
    src.mkdir()
    _meas_dir, db_path = _save_measurements_and_well_summary(
        df, well_df, str(src), "motility"
    )

    con = sqlite3.connect(db_path)
    try:
        got = pd.read_sql("SELECT * FROM motility_well_motility", con)
    finally:
        con.close()
    assert sorted(got["wellID"].tolist()) == ["A01", "A02"]
    assert set(got["velocity_unit"]) == {unit}
    assert got.set_index("wellID").loc["A02", "mean_velocity_all"] == pytest.approx(6.0)
