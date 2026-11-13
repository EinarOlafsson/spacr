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
# _apply_infection_intensity_qc with stand-in strategy helpers
# ---------------------------------------------------------------------------

def _two_plate_frame():
    a = _qc_frame(n_cells=4, n_frames=1, plate="plateA", well="A01")
    b = _qc_frame(n_cells=4, n_frames=1, plate="plateB", well="B02", cell_offset=100)
    return pd.concat([a, b], ignore_index=True)


def _three_well_frame():
    return pd.concat(
        [
            _qc_frame(n_cells=2, n_frames=1, plate="plateA", well="A01"),
            _qc_frame(n_cells=2, n_frames=1, plate="plateA", well="A02",
                      cell_offset=50),
            _qc_frame(n_cells=2, n_frames=1, plate="plateB", well="A01",
                      cell_offset=90),
        ],
        ignore_index=True,
    )


def _ghost_plate_frame():
    df = _qc_frame(n_cells=4, n_frames=1, plate="plateA", well="A01")
    df["plateID"] = pd.Categorical(df["plateID"], categories=["plateA", "plateGHOST"])
    return df


def _plain4():
    return _qc_frame(n_cells=4, n_frames=1)


def _small_frame_no(col):
    return lambda: _qc_frame(n_cells=4, n_frames=1).drop(columns=[col])


_STALE_ALL = {k: "stale" for k in _PAYLOAD_KEYS}
_ON = {"infection_intensity_qc": True}


def _on(**kw):
    return {**_ON, **kw}


_ALTERNATING = [0, 1, 0, 1]   # row 0 was NaN with label False -> 0, then 0/1 pattern

_QC_ROWS = [
    # no-op paths
    pytest.param("passthrough", _plain4, {"infection_intensity_qc": False, **_STALE_ALL}, 2,
                 dict(calls_names=[], out_is_df=True, col="infected",
                      reset_keys=_PAYLOAD_KEYS, dir_exists=False), id="qc-disabled-resets-payloads"),
    pytest.param("passthrough", _plain4, _on(), None,
                 dict(calls_names=[], out_is_df=True, col="infected"), id="no-pathogen-channel"),
] + [
    pytest.param("passthrough", _plain4,
                 _on(infection_intensity_qc_scope=scope, infection_hist_data="stale"), 2,
                 dict(calls_names=[], out_is_df=True, col="infected",
                      reset_keys=("infection_hist_data",), dir_exists=True),
                 id=f"scope-none-{scope}")
    for scope in ["none", "off", "NONE", "Off"]
] + [
    pytest.param("passthrough", _plain4,
                 _on(infection_intensity_strategy=strategy,
                     infection_intensity_qc_scope="combined"), 3,
                 dict(calls_names=[expected], col="infected", adjusted=None, len_out=4,
                      dispatch=3, dir_exists=True,
                      payloads={"infection_intensity_qc_panel_type": expected,
                                "infection_hist_data": "hist::plate1"}),
                 id=f"strategy-{strategy}")
    for strategy, expected in [
        ("hist", "histogram"), ("histogram", "histogram"), ("HISTAGRAM", "histogram"),
        ("xgboost", "xgboost"), ("XGB", "xgboost"), ("pca", "pca"), ("umap", "pca"),
        ("tsne", "pca"), ("banana", "histogram"), (None, "histogram"),
    ]
] + [
    pytest.param("passthrough", _plain4, _on(), 1,
                 dict(calls_names=["histogram"]), id="default-strategy-is-histogram"),
] + [
    # single global run
    pytest.param("nan_numeric", build, _on(infection_intensity_strategy="hist",
                                           infection_intensity_qc_scope=scope), 2,
                 dict(calls_names=["histogram"], n_rows=[4], col="adjusted_infected",
                      adjusted="int", values=_ALTERNATING, xgb_tag=True),
                 id=f"global-fills-nan-and-casts-int-{case}")
    for case, scope, build in [
        ("combined", "combined", _plain4),
        ("global", "GLOBAL", _plain4),
        ("all", "all", _plain4),
        ("unknown", "sideways", _plain4),
        ("missing-plate-col", "plate", _small_frame_no("plateID")),
        ("missing-well-col", "per_well", _small_frame_no("wellID")),
    ]
] + [
    pytest.param("strings", build, _on(infection_intensity_strategy="hist",
                                       infection_intensity_qc_scope=scope), 2,
                 dict(calls_names=["histogram"], col="adjusted_infected",
                      adjusted="bool_all"),
                 id=f"global-falls-back-to-bool-cast-{case}")
    for case, scope, build in [
        ("combined", "combined", _plain4),
        ("unknown", "sideways", _plain4),
        ("missing-plate-col", "plate", _small_frame_no("plateID")),
    ]
] + [
    pytest.param("passthrough", _plain4, _on(infection_intensity_qc_scope="combined"), 2,
                 dict(calls_names=["histogram"], col="infected", adjusted=None,
                      out_is_df=False), id="global-without-adjusted-column-keeps-helper-column"),
    pytest.param("passthrough", _plain4, _on(infection_intensity_qc_scope=None), 2,
                 dict(calls_names=["histogram"], n_rows=[4], col="infected"),
                 id="scope-none-value-defaults-to-combined"),
] + [
    # grouped scopes
    pytest.param("nan_numeric", _two_plate_frame, _on(infection_intensity_qc_scope=scope), 2,
                 dict(calls_tags=["plateA", "plateB"], n_rows=[4, 4], col="adjusted_infected",
                      len_out=8, adjusted="int",
                      payloads={"infection_hist_data": "hist::plateA",
                                "infection_pca_data": "pca::plateA",
                                "infection_intensity_qc_panel_path": "histogram::plateA.png"}),
                 id=f"scope-plate-runs-per-plate-{scope}")
    for scope in ["plate", "PER_PLATE", "plateid"]
] + [
    pytest.param("passthrough", _three_well_frame, _on(infection_intensity_qc_scope=scope), 2,
                 dict(n_calls=3, n_rows=[2, 2, 2], col="infected", len_out=6,
                      wells=["A01", "A02"]),
                 id=f"scope-well-runs-per-well-{scope}")
    for scope in ["well", "per_well"]
] + [
    pytest.param("nan_numeric", _ghost_plate_frame, _on(infection_intensity_qc_scope="plate"), 2,
                 dict(calls_tags=["plateA"], len_out=4, col="adjusted_infected"),
                 id="scope-group-skips-empty-categorical-groups"),
    pytest.param("nan_numeric", lambda: _qc_frame(n_cells=2, n_frames=1).iloc[0:0],
                 _on(infection_intensity_qc_scope="well"), 2,
                 dict(calls_names=[], out_is_df=True, col="infected", empty_in=True),
                 id="scope-group-with-empty-frame-returns-input"),
    pytest.param("all_nan", _two_plate_frame, _on(infection_intensity_qc_scope="plate"), 2,
                 dict(n_calls=2, col="infected", adjusted="all_nan"),
                 id="scope-group-all-nan-adjusted-keeps-original-column"),
    pytest.param("strings", _two_plate_frame, _on(infection_intensity_qc_scope="plate"), 2,
                 dict(col="adjusted_infected", adjusted="bool_all", len_out=8),
                 id="scope-group-bool-cast-fallback"),
]


@pytest.mark.parametrize("mode, build, settings, chan, expect", _QC_ROWS)
def test_infection_qc_dispatches_strategy_and_scope_and_adjusts_labels(
        tmp_path, monkeypatch, mode, build, settings, chan, expect):
    """Strategy choice, scope grouping, no-op paths and the label cast of the QC dispatcher."""
    from spacr.timelapse import _apply_infection_intensity_qc

    calls = []
    _patch_all_strategies(monkeypatch, calls, mode=mode)
    df = build()
    settings = dict(settings)
    motility_dir = str(tmp_path / "mot")
    if expect.get("empty_in"):
        assert df.empty and "plateID" in df.columns

    out, col = _apply_infection_intensity_qc(df, settings, "infected", chan, motility_dir)

    if "calls_names" in expect:
        assert [c["name"] for c in calls] == expect["calls_names"]
    if "calls_tags" in expect:
        assert [c["tag"] for c in calls] == expect["calls_tags"]
    if "n_calls" in expect:
        assert len(calls) == expect["n_calls"]
    if "n_rows" in expect:
        assert [c["n_rows"] for c in calls] == expect["n_rows"]
        if len(expect["n_rows"]) == 1:
            assert calls[0]["n_rows"] == len(df)
    if "col" in expect:
        assert col == expect["col"]
    if expect.get("out_is_df") is True:
        assert out is df
    if expect.get("out_is_df") is False:
        assert out is not df
    if "len_out" in expect:
        assert len(out) == len(df) == expect["len_out"]
    if "dir_exists" in expect:
        assert os.path.exists(motility_dir) is expect["dir_exists"]
    for key in expect.get("reset_keys", ()):
        assert settings[key] is None
    for key, value in expect.get("payloads", {}).items():
        assert settings[key] == value
    if "dispatch" in expect:
        assert calls[0]["pathogen_chan"] == expect["dispatch"]
        assert calls[0]["motility_dir"] == motility_dir
        assert calls[0]["infection_col"] == "infected"
        # the helper is handed a COPY of settings, not the caller dict
        assert calls[0]["settings"] is not settings
    if expect.get("xgb_tag"):
        assert settings["infection_xgb_importance"] == "xgb::" + calls[0]["tag"]
    if "wells" in expect:
        assert sorted(out["wellID"].unique()) == expect["wells"]

    kind = expect.get("adjusted", "skip")
    if kind is None:
        assert "adjusted_infected" not in out.columns
    elif kind == "int":
        assert not out["adjusted_infected"].isna().any()
        assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
        if "values" in expect:
            assert out["adjusted_infected"].tolist() == expect["values"]
    elif kind == "bool_all":
        assert out["adjusted_infected"].dtype == bool
        assert out["adjusted_infected"].all()
    elif kind == "all_nan":
        assert "adjusted_infected" in out.columns
        assert out["adjusted_infected"].isna().all()


# ---------------------------------------------------------------------------
# real histogram strategy, end to end through the dispatcher
# ---------------------------------------------------------------------------

def _real_combined(tmp_path):
    from spacr.timelapse import _apply_infection_intensity_qc

    df = _qc_frame(n_cells=24, n_frames=2, pathogen_chan=2)
    motility_dir = str(tmp_path / "motility")
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_qc_scope": "combined",
        "infection_intensity_qc_graphs": True,
    }
    out, col = _apply_infection_intensity_qc(df, settings, "infected", 2, motility_dir)
    assert col == "adjusted_infected"
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    # bright cells (second half) are called infected, dim ones are not
    bright = out["cellID"] > 12
    assert out.loc[bright, "adjusted_infected"].eq(1).all()
    assert out.loc[~bright, "adjusted_infected"].eq(0).all()
    payload = settings["infection_hist_data"]
    assert isinstance(payload, dict)
    assert payload["intensity_col"] == "cell_p95_intensity_ch2"
    # threshold must sit strictly between the dim (<=131) and bright (>=1112) populations
    assert 131.0 < payload["thr_val"] <= 1112.0
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    png = settings["infection_intensity_qc_panel_path"]
    assert png is not None and os.path.isfile(png)
    assert os.path.dirname(png) == motility_dir
    return {
        "col": col, "bright_sum": int(out.loc[bright, "adjusted_infected"].sum()),
        "n_bright": int(bright.sum()), "dim_sum": int(out.loc[~bright, "adjusted_infected"].sum()),
        "intensity_col": payload["intensity_col"], "panel_type": settings["infection_intensity_qc_panel_type"],
        "png_dir_is_motility_dir": os.path.dirname(png) == motility_dir,
        "threshold_between_populations": bool(131.0 < payload["thr_val"] <= 1112.0),
    }


def _real_per_plate(tmp_path):
    from spacr.timelapse import _apply_infection_intensity_qc

    df = pd.concat(
        [
            _qc_frame(n_cells=24, n_frames=1, plate="plateA", well="A01"),
            _qc_frame(n_cells=24, n_frames=1, plate="plateB", well="B01", cell_offset=100),
        ],
        ignore_index=True,
    )
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "hist",
        "infection_intensity_qc_scope": "plate",
        "infection_intensity_qc_graphs": False,   # keep it fast: no PNG
    }
    out, col = _apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "motility")
    )
    assert col == "adjusted_infected"
    assert len(out) == len(df)
    assert pd.api.types.is_integer_dtype(out["adjusted_infected"])
    # each plate was thresholded on its own, both recover the injected split
    for plate in ("plateA", "plateB"):
        sub = out[out["plateID"] == plate]
        assert sub["adjusted_infected"].sum() == 12
        assert sub.loc[sub["infected"], "adjusted_infected"].eq(1).all()
    # graphs disabled -> payload present but no PNG on disk
    assert settings["infection_intensity_qc_panel_path"] is None
    assert isinstance(settings["infection_hist_data"], dict)
    return {
        "col": col, "n_rows": len(out),
        "infected_per_plate": {pl: int(out.loc[out["plateID"] == pl, "adjusted_infected"].sum())
                               for pl in ("plateA", "plateB")},
        "panel_path": settings["infection_intensity_qc_panel_path"],
        "hist_payload_is_dict": isinstance(settings["infection_hist_data"], dict),
    }


@pytest.mark.parametrize("run, expected", [
    pytest.param(_real_combined, {
        "col": "adjusted_infected", "bright_sum": 24, "n_bright": 24, "dim_sum": 0,
        "intensity_col": "cell_p95_intensity_ch2", "panel_type": "histogram",
        "png_dir_is_motility_dir": True, "threshold_between_populations": True,
    }, id="combined-scope-with-graph"),
    pytest.param(_real_per_plate, {
        "col": "adjusted_infected", "n_rows": 48,
        "infected_per_plate": {"plateA": 12, "plateB": 12},
        "panel_path": None, "hist_payload_is_dict": True,
    }, id="per-plate-scope-without-graph"),
])
def test_infection_qc_real_histogram_strategy_recovers_the_injected_split(tmp_path, run, expected):
    observed = run(tmp_path)
    assert observed == expected


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


def _any_infected_frame():
    rows = _track_rows("plate1", "A01", 1, 1, [0, 2, 4], [0, 0, 0], False)
    rows[1]["infected"] = True
    return pd.DataFrame(rows)


def _chk_missing_centroids(track_df, per_well, well_df, unit):
    assert track_df.empty and well_df.empty
    assert per_well == {}
    assert unit == "px/frame"


def _chk_physical_units(track_df, per_well, well_df, unit):
    assert unit == "µm/min"
    assert len(track_df) == 1
    rec = track_df.iloc[0]
    assert rec["v_px_per_frame"] == pytest.approx(3.0)
    assert rec["velocity"] == pytest.approx(2.0)
    assert rec["straightness"] == pytest.approx(1.0)
    assert bool(rec["infected"]) is True
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


def _chk_pixel_units(track_df, per_well, well_df, unit):
    assert unit == "px/frame"
    assert track_df.iloc[0]["velocity"] == pytest.approx(4.0)
    assert well_df.iloc[0]["n_uninfected_tracks"] == 1
    assert well_df.iloc[0]["n_infected_tracks"] == 0
    assert np.isnan(well_df.iloc[0]["mean_velocity_infected"])
    assert well_df.iloc[0]["mean_velocity_uninfected"] == pytest.approx(4.0)


def _chk_short_and_nonfinite(track_df, per_well, well_df, unit):
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


def _chk_no_usable_tracks(track_df, per_well, well_df, unit):
    assert track_df.empty and well_df.empty
    assert per_well == {}
    assert unit == "px/frame"          # never upgraded to physical units


def _chk_filter_prunes_wells(track_df, per_well, well_df, unit):
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


def _chk_filter_removes_every_track(track_df, per_well, well_df, unit):
    assert track_df.empty
    assert per_well == {}
    assert well_df.empty
    # units were already resolved before the filter emptied the table
    assert unit == "µm/min"


def _chk_filter_off_keeps_straight(track_df, per_well, well_df, unit):
    assert len(track_df) == 1
    assert track_df.iloc[0]["straightness"] == pytest.approx(1.0)
    assert len(per_well[("plate1", "A01")]) == 1
    assert well_df.iloc[0]["n_tracks"] == 1


def _chk_mixed_means(track_df, per_well, well_df, unit):
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


def _chk_any_frame_infected(track_df, per_well, well_df, unit):
    # infected on a single frame only -> the whole track counts as infected
    assert bool(track_df.iloc[0]["infected"]) is True
    assert well_df.iloc[0]["n_infected_tracks"] == 1
    assert well_df.iloc[0]["n_uninfected_tracks"] == 0


_NAN = float("nan")
_STRAIGHT = [("plate1", "A01", 1, 1, [0, 3, 6, 9], [0, 0, 0, 0], True)]
_FILTER = {"drop_straight_tracks": True, "straightness_threshold": 0.95}

_VELOCITY_ROWS = [
    pytest.param(lambda: _qc_frame(n_cells=2, n_frames=2), {}, 1.0, 60.0,
                 _chk_missing_centroids, ("px/frame", 0, 0, {}, pytest.approx([]), pytest.approx([], nan_ok=True)), id="missing-centroid-columns"),
    pytest.param(lambda: _tracks_frame(_STRAIGHT), {}, 3.0, 30.0,
                 _chk_physical_units, ("µm/min", 1, 1, {("plate1", "A01"): 1}, pytest.approx([2.0]), pytest.approx([1.0], nan_ok=True)), id="physical-units-and-straightness"),
] + [
    pytest.param(lambda: _tracks_frame([("plate1", "A01", 1, 1, [0, 4, 8], [0, 0, 0], False)]),
                 {}, ppu, spf, _chk_pixel_units, ("px/frame", 1, 1, {("plate1", "A01"): 1}, pytest.approx([4.0]), pytest.approx([1.0], nan_ok=True)), id=f"pixel-units-when-calibration-missing-{name}")
    for name, ppu, spf in [("no-scale", None, 30.0), ("no-interval", 3.0, None),
                           ("neither", None, None)]
] + [
    pytest.param(lambda: _tracks_frame([
        ("plate1", "A01", 1, 1, [0], [0], False),
        ("plate1", "A01", 1, 2, [_NAN] * 3, [_NAN] * 3, False),
        ("plate1", "A01", 1, 3, [5, 5, 5], [7, 7, 7], False),
        ("plate1", "A01", 1, 4, [0, 2, 4], [0, 0, 0], True),
    ]), {}, None, None, _chk_short_and_nonfinite, ("px/frame", 2, 1, {("plate1", "A01"): 2}, pytest.approx([0.0, 2.0]), pytest.approx([float("nan"), 1.0], nan_ok=True)), id="skips-short-and-nonfinite-tracks"),
    pytest.param(lambda: _tracks_frame([
        ("plate1", "A01", 1, 1, [0], [0], False),
        ("plate1", "A01", 1, 2, [3], [3], True),
    ]), {}, 2.0, 60.0, _chk_no_usable_tracks, ("px/frame", 0, 0, {}, pytest.approx([]), pytest.approx([], nan_ok=True)), id="no-usable-tracks-returns-empty"),
    pytest.param(lambda: _tracks_frame([
        ("plate1", "A01", 1, 1, [0, 3, 6, 9], [0, 0, 0, 0], True),
        ("plate1", "A02", 1, 2, [0, 3, 6, 9], [0, 0, 0, 0], True),
        ("plate1", "A02", 1, 3, [0, 5, 0, 5], [0, 0, 0, 0], False),
    ]), _FILTER, None, None, _chk_filter_prunes_wells, ("px/frame", 1, 1, {("plate1", "A02"): 1}, pytest.approx([5.0]), pytest.approx([1 / 3], nan_ok=True)),
        id="straightness-filter-drops-tracks-and-prunes-wells"),
    pytest.param(lambda: _tracks_frame([
        ("plate1", "A01", 1, 1, [0, 3, 6], [0, 0, 0], True),
        ("plate1", "A02", 1, 2, [0, 1, 2], [0, 0, 0], False),
    ]), {"drop_straight_tracks": True, "straightness_threshold": 0.5}, 1.0, 60.0,
        _chk_filter_removes_every_track, ("µm/min", 0, 0, {}, pytest.approx([]), pytest.approx([], nan_ok=True)), id="straightness-filter-can-remove-every-track"),
    pytest.param(lambda: _tracks_frame([("plate1", "A01", 1, 1, [0, 3, 6], [0, 0, 0], True)]),
                 {"straightness_threshold": 0.9}, None, None, _chk_filter_off_keeps_straight, ("px/frame", 1, 1, {("plate1", "A01"): 1}, pytest.approx([3.0]), pytest.approx([1.0], nan_ok=True)),
                 id="straightness-filter-off-keeps-straight-tracks"),
    pytest.param(lambda: _tracks_frame([
        ("plate1", "A01", 1, 1, [0, 5, 0, 5], [0, 0, 0, 0], True),
        ("plate1", "A01", 2, 2, [0, 1, 0, 1], [0, 0, 0, 0], False),
    ]), {}, None, None, _chk_mixed_means, ("px/frame", 2, 1, {("plate1", "A01"): 2}, pytest.approx([5.0, 1.0]), pytest.approx([1 / 3, 1 / 3], nan_ok=True)), id="well-summary-mixed-infection-means"),
    pytest.param(_any_infected_frame, {}, None, None, _chk_any_frame_infected, ("px/frame", 1, 1, {("plate1", "A01"): 1}, pytest.approx([2.0]), pytest.approx([1.0], nan_ok=True)),
                 id="any-infected-frame-marks-track-infected"),
]


@pytest.mark.parametrize("build, settings, ppu, spf, check, expected", _VELOCITY_ROWS)
def test_velocities_and_well_summary_report_per_track_and_per_well_motility(
        build, settings, ppu, spf, check, expected):
    from spacr.timelapse import _compute_velocities_and_well_summary

    out = _compute_velocities_and_well_summary(build(), dict(settings), "infected", ppu, spf)
    track_df, per_well, well_df, unit = out
    check(*out)
    observed = (
        unit, len(track_df), len(well_df),
        {k: len(v) for k, v in per_well.items()},
        track_df["velocity"].tolist() if "velocity" in track_df else [],
        track_df["straightness"].tolist() if "straightness" in track_df else [],
    )
    assert observed == expected


# ---------------------------------------------------------------------------
# _save_measurements_and_well_summary
# ---------------------------------------------------------------------------

def _tables(con):
    return {
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _save_with_well_summary(src):
    from spacr.timelapse import _save_measurements_and_well_summary

    all_df = pd.DataFrame(
        {"plateID": ["p1", "p1"], "wellID": ["A01", "A02"], "value": [1.5, 2.5]}
    )
    well_df = pd.DataFrame(
        {"plateID": ["p1"], "wellID": ["A01"], "n_tracks": [3], "velocity_unit": ["px/frame"]}
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
        assert {"motility", "motility_well_motility"} <= _tables(con)
        assert pd.read_sql("SELECT * FROM motility", con)["value"].tolist() == [1.5, 2.5]
        got_well = pd.read_sql("SELECT * FROM motility_well_motility", con)
        assert got_well["n_tracks"].tolist() == [3]
        assert got_well["velocity_unit"].tolist() == ["px/frame"]
        observed = (os.path.basename(meas_dir), os.path.basename(db_path),
                    sorted(_tables(con)), pd.read_sql("SELECT * FROM motility", con)["value"].tolist(),
                    got_well["n_tracks"].tolist(), got_well["velocity_unit"].tolist())
    finally:
        con.close()
    return observed


def _save_without_well_summary(src):
    from spacr.timelapse import _save_measurements_and_well_summary

    (src / "measurements").mkdir(parents=True)   # already-existing dir is fine
    meas_dir, db_path = _save_measurements_and_well_summary(
        pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame(), str(src), "tracks"
    )
    assert os.path.isfile(db_path)
    con = sqlite3.connect(db_path)
    try:
        tables = _tables(con)
        assert "tracks" in tables
        assert "tracks_well_motility" not in tables
        assert pd.read_sql("SELECT * FROM tracks", con)["a"].tolist() == [1, 2, 3]
        observed = (sorted(tables), pd.read_sql("SELECT * FROM tracks", con)["a"].tolist())
    finally:
        con.close()
    assert meas_dir.endswith("measurements")
    return observed + (os.path.basename(meas_dir),)


def _save_replaces_existing_table(src):
    from spacr.timelapse import _save_measurements_and_well_summary

    _save_measurements_and_well_summary(
        pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame(), str(src), "tracks"
    )
    _, db_path = _save_measurements_and_well_summary(
        pd.DataFrame({"a": [9]}), pd.DataFrame(), str(src), "tracks"
    )
    con = sqlite3.connect(db_path)
    try:
        got = pd.read_sql("SELECT * FROM tracks", con)
    finally:
        con.close()
    assert got["a"].tolist() == [9]
    return got["a"].tolist()


def _save_roundtrips_velocity_output(src):
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
    return (sorted(got["wellID"].tolist()), sorted(set(got["velocity_unit"])),
            float(got.set_index("wellID").loc["A02", "mean_velocity_all"]))


@pytest.mark.parametrize("scenario, expected", [
    pytest.param(_save_with_well_summary,
                 ("measurements", "measurements.db", ["motility", "motility_well_motility"],
                  [1.5, 2.5], [3], ["px/frame"]), id="with-well-summary"),
    pytest.param(_save_without_well_summary,
                 (["tracks"], [1, 2, 3], "measurements"), id="without-well-summary"),
    pytest.param(_save_replaces_existing_table, [9], id="replaces-existing-table"),
    pytest.param(_save_roundtrips_velocity_output,
                 (["A01", "A02"], ["px/frame"], pytest.approx(6.0)), id="roundtrips-velocity-output"),
])
def test_save_measurements_writes_tables_into_the_project_database(tmp_path, scenario, expected):
    src = tmp_path / "proj"
    if scenario is not _save_without_well_summary:
        src.mkdir()
    observed = scenario(src)
    assert observed == expected
