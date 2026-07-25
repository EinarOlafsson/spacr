"""CPU coverage for the infection-QC panel helpers in ``spacr.timelapse``.

Covers four private helpers used by ``summarise_tracks_from_merged``:

* ``_compute_intensity_percentiles_per_channel`` -- per-frame / per-object
  intensity percentiles, including all of its early-return guards.
* ``_make_adjusted_qc_panel`` -- the 3-panel PNG (PCA / XGBoost / histogram)
  built purely from payloads stashed in ``settings``.
* ``_load_measurements_from_db`` -- SQLite reader with its failure paths.
* ``_infection_qc_histogram`` -- histogram-based infection relabelling
  (relabel / remove modes, log transform, threshold fallback, graph on/off).

Everything is synthetic, offline and Agg-only.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _qc_df(n_cells=30, frames=2, chan=2, col="p95", seed=0,
           plate="plate1", well="A01"):
    """Per-frame per-cell table with a clean infected / uninfected split.

    Uninfected cells sit around 100-200 intensity, infected around 5000-9000,
    so the histogram threshold search has an obvious answer.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cells):
        infected = i >= n_cells // 2
        base = 5000.0 + 4000.0 * rng.random() if infected else 100.0 + 80.0 * rng.random()
        for f in range(frames):
            rows.append(
                {
                    "plateID": plate,
                    "wellID": well,
                    "fieldID": 1,
                    "cellID": i + 1,
                    "frame": f,
                    "infected": bool(infected),
                    f"cell_{col}_intensity_ch{chan}": float(base + rng.random()),
                }
            )
    return pd.DataFrame(rows)


def _per_cell_positive(df, col="adjusted_infected"):
    """Number of distinct cells whose label is True."""
    return int(df.groupby("cellID")[col].max().astype(bool).sum())


# ---------------------------------------------------------------------------
# _compute_intensity_percentiles_per_channel
# ---------------------------------------------------------------------------

def _labelled_stack(T=3, H=16, W=16):
    mask = np.zeros((T, H, W), dtype=np.int32)
    for t in range(T):
        mask[t, 2:6, 2:6] = 1
        mask[t, 9:14, 9:14] = 2
    return mask


def test_percentiles_normal_path_values_and_columns():
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=2)
    inten = np.zeros((2, 16, 16, 3), dtype=float)
    # Channel 1 gets a deterministic ramp so percentiles are predictable.
    inten[:, :, :, 1] = 7.0
    inten[0, 2:6, 2:6, 1] = np.arange(16, dtype=float).reshape(4, 4)
    inten[0, 9:14, 9:14, 1] = 100.0

    out = _compute_intensity_percentiles_per_channel(
        mask, inten, 1, "cell", percentiles=(25, 50, 75)
    )

    assert list(out.columns) == [
        "frame",
        "cell_label",
        "cell_p25_intensity_ch1",
        "cell_p50_intensity_ch1",
        "cell_p75_intensity_ch1",
    ]
    # 2 frames x 2 objects
    assert len(out) == 4
    assert set(out["cell_label"]) == {1, 2}
    assert set(out["frame"]) == {0, 1}

    r = out[(out["frame"] == 0) & (out["cell_label"] == 1)].iloc[0]
    assert r["cell_p50_intensity_ch1"] == pytest.approx(7.5)
    assert r["cell_p25_intensity_ch1"] == pytest.approx(np.percentile(np.arange(16.0), 25))

    r2 = out[(out["frame"] == 0) & (out["cell_label"] == 2)].iloc[0]
    assert r2["cell_p75_intensity_ch1"] == pytest.approx(100.0)
    # frame 1 is the flat 7.0 background inside both labels
    r3 = out[out["frame"] == 1].iloc[0]
    assert r3["cell_p50_intensity_ch1"] == pytest.approx(7.0)
    assert out["cell_label"].dtype.kind in "iu"


def test_percentiles_label_as_track_id_renames_column():
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 2), 3.0)

    out = _compute_intensity_percentiles_per_channel(
        mask, inten, 0, "cell", percentiles=(50,), label_as_track_id=True
    )
    assert "track_id" in out.columns
    assert "cell_label" not in out.columns
    assert out["cell_p50_intensity_ch0"].tolist() == [3.0, 3.0]


def test_percentiles_none_intensity_stack_returns_empty_frame():
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    out = _compute_intensity_percentiles_per_channel(
        _labelled_stack(), None, 0, "nucleus"
    )
    assert out.empty
    assert list(out.columns) == ["frame", "nucleus_label"]

    out_tid = _compute_intensity_percentiles_per_channel(
        _labelled_stack(), None, 0, "nucleus", label_as_track_id=True
    )
    assert list(out_tid.columns) == ["frame", "track_id"]


@pytest.mark.parametrize("chan", [None, -1, 5])
def test_percentiles_bad_channel_index_returns_empty_frame(chan):
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.ones((1, 16, 16, 3), dtype=float)
    out = _compute_intensity_percentiles_per_channel(mask, inten, chan, "pathogen")
    assert out.empty
    assert list(out.columns) == ["frame", "pathogen_label"]


def test_percentiles_skips_empty_frames_and_returns_empty_when_all_empty():
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = np.zeros((3, 8, 8), dtype=np.int32)
    mask[1, 1:4, 1:4] = 4  # only frame 1 has an object
    inten = np.full((3, 8, 8, 1), 2.5)

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cytoplasm",
                                                     percentiles=(50,))
    assert len(out) == 1
    assert out["frame"].tolist() == [1]
    assert out["cytoplasm_label"].tolist() == [4]

    # No labels at all anywhere -> the "no dfs" return.
    empty = _compute_intensity_percentiles_per_channel(
        np.zeros((2, 8, 8), dtype=np.int32), inten[:2], 0, "cell"
    )
    assert empty.empty
    assert list(empty.columns) == ["frame", "cell_label"]


def test_percentiles_negative_only_labels_are_skipped():
    """np.any() is truthy for a -1 mask but there are no labels > 0."""
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = np.zeros((1, 8, 8), dtype=np.int32)
    mask[0, 0:3, 0:3] = -1
    inten = np.ones((1, 8, 8, 1), dtype=float)

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell")
    assert out.empty
    assert list(out.columns) == ["frame", "cell_label"]


def test_percentiles_all_nan_object_is_dropped():
    """A label whose pixels are all NaN yields no record (vals.size == 0)."""
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 1), 5.0)
    inten[0, 2:6, 2:6, 0] = np.nan  # label 1 is entirely non-finite

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell",
                                                     percentiles=(50,))
    assert out["cell_label"].tolist() == [2]
    assert out["cell_p50_intensity_ch0"].tolist() == [5.0]


def test_percentiles_all_objects_nan_returns_empty():
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 1), np.nan)
    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell")
    assert out.empty
    assert list(out.columns) == ["frame", "cell_label"]


# ---------------------------------------------------------------------------
# _load_measurements_from_db
# ---------------------------------------------------------------------------

def test_load_measurements_missing_file_returns_empty(tmp_path):
    from spacr.timelapse import _load_measurements_from_db

    out = _load_measurements_from_db(str(tmp_path / "nope.db"), "cell")
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert list(out.columns) == []


def test_load_measurements_reads_rows(tmp_path):
    from spacr.timelapse import _load_measurements_from_db

    db = tmp_path / "measurements.db"
    src = pd.DataFrame({"cellID": [1, 2, 3], "cell_area": [10.0, 20.0, 30.0]})
    con = sqlite3.connect(db)
    try:
        src.to_sql("tracked_cells", con, index=False)
    finally:
        con.close()

    out = _load_measurements_from_db(str(db), "tracked_cells")
    assert list(out.columns) == ["cellID", "cell_area"]
    assert out["cell_area"].tolist() == [10.0, 20.0, 30.0]


def test_load_measurements_missing_table_returns_empty(tmp_path, capsys):
    from spacr.timelapse import _load_measurements_from_db

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    try:
        pd.DataFrame({"a": [1]}).to_sql("present", con, index=False)
    finally:
        con.close()

    out = _load_measurements_from_db(str(db), "absent_table")
    assert out.empty
    assert "Could not load existing measurements" in capsys.readouterr().out


def test_load_measurements_unreadable_file_returns_empty(tmp_path, capsys):
    """A non-SQLite file still opens but every query raises."""
    from spacr.timelapse import _load_measurements_from_db

    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is definitely not a sqlite database" * 8)

    out = _load_measurements_from_db(str(junk), "cell")
    assert out.empty
    assert "Could not load existing measurements" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _make_adjusted_qc_panel
# ---------------------------------------------------------------------------

def _panel_df(plate="plate1", well="A01"):
    return pd.DataFrame({"plateID": [plate] * 4, "wellID": [well] * 4})


def _full_payload(rng=None, log=False, thr=1.5, chan=2):
    rng = rng or np.random.default_rng(1)
    return {
        "infection_hist_data": {
            "intensities_inf": rng.normal(5.0, 1.0, 40),
            "intensities_uninf": rng.normal(1.0, 0.5, 40),
            "bin_edges": np.linspace(-2, 9, 21),
            "thr_val": thr,
            "pathogen_chan": chan,
            "log_transform": log,
        },
        "infection_pca_data": {
            "coords": rng.normal(size=(30, 3)),
            "labels": np.array([True, False] * 15),
            "method_label": "UMAP",
        },
        "infection_xgb_importance": {
            "feature_names": ["cell_area", "cell_p95_intensity_ch2"],
            "feature_importances": [0.7, 0.3],
        },
    }


def test_adjusted_qc_panel_full_payload_writes_png(tmp_path):
    from spacr.timelapse import _make_adjusted_qc_panel

    out_dir = tmp_path / "motility"  # deliberately absent -> exercises makedirs
    settings = _full_payload()
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(out_dir),
                            settings, "adjusted")

    expected = out_dir / "infection_qc_panel_adjusted_plate1_A01.png"
    assert expected.is_file()
    assert expected.stat().st_size > 1000
    assert expected.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_adjusted_qc_panel_log_and_multi_well_tag(tmp_path, capsys):
    from spacr.timelapse import _make_adjusted_qc_panel

    df = pd.DataFrame(
        {"plateID": ["plate1"] * 4, "wellID": ["A01", "A01", "B02", "B02"]}
    )
    settings = _full_payload(log=True)
    _make_adjusted_qc_panel(df, "adjusted_infected", str(tmp_path), settings, "mask")

    expected = tmp_path / "infection_qc_panel_mask_plate1_MULTI_WELLS.png"
    assert expected.is_file()
    assert str(expected) in capsys.readouterr().out


def test_adjusted_qc_panel_no_threshold_and_no_channel(tmp_path):
    """thr_val None skips the axvline, pathogen_chan None uses the plain label."""
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = _full_payload(thr=None, chan=None)
    settings["infection_hist_data"]["thr_val"] = None
    settings["infection_hist_data"]["pathogen_chan"] = None
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    assert (tmp_path / "infection_qc_panel_adjusted_plate1_A01.png").is_file()


def test_adjusted_qc_panel_empty_payloads_render_placeholders(tmp_path):
    """No hist / no PCA / no XGB -> all three 'no data' placeholder branches."""
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = {}
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    assert (tmp_path / "infection_qc_panel_adjusted_plate1_A01.png").is_file()


def test_adjusted_qc_panel_malformed_payloads_render_placeholders(tmp_path):
    """Present-but-unusable payloads take the same placeholder branches."""
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = {
        # values present but no bins -> histogram placeholder
        "infection_hist_data": {
            "intensities_inf": [1.0, 2.0],
            "intensities_uninf": [0.1],
            "bin_edges": [],
        },
        # coords is 1-D -> shape check fails -> PCA placeholder
        "infection_pca_data": {
            "coords": np.arange(10.0),
            "labels": np.zeros(10, dtype=bool),
        },
        # name/value length mismatch -> XGB placeholder
        "infection_xgb_importance": {
            "feature_names": ["a", "b", "c"],
            "feature_importances": [1.0],
        },
    }
    _make_adjusted_qc_panel(_panel_df(plate="p9", well="H12"),
                            "adjusted_infected", str(tmp_path), settings, "xgb")
    assert (tmp_path / "infection_qc_panel_xgb_p9_H12.png").is_file()


def test_adjusted_qc_panel_pca_labels_none_takes_placeholder(tmp_path):
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = _full_payload()
    settings["infection_pca_data"] = {"coords": np.zeros((5, 2)), "labels": None}
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    assert (tmp_path / "infection_qc_panel_adjusted_plate1_A01.png").is_file()


def test_adjusted_qc_panel_meta_tag_without_plate_columns(tmp_path):
    """No plateID/wellID at all -> MULTI_PLATES_MULTI_WELLS filename."""
    from spacr.timelapse import _make_adjusted_qc_panel

    _make_adjusted_qc_panel(pd.DataFrame({"cellID": [1, 2]}), "adjusted_infected",
                            str(tmp_path), _full_payload(), "adjusted")
    assert (tmp_path /
            "infection_qc_panel_adjusted_MULTI_PLATES_MULTI_WELLS.png").is_file()


# ---------------------------------------------------------------------------
# _infection_qc_histogram
# ---------------------------------------------------------------------------

def test_infection_qc_histogram_relabel_writes_png_and_payload(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert col == "adjusted_infected"
    assert "adjusted_infected" in out.columns
    assert len(out) == len(df)
    # The synthetic split is clean, so the histogram must reproduce it exactly.
    assert _per_cell_positive(out) == 15
    assert out["adjusted_infected"].dtype == bool

    png = tmp_path / "infection_intensity_histogram_plate1_A01.png"
    assert png.is_file()
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    assert settings["infection_intensity_qc_panel_path"] == str(png)

    payload = settings["infection_hist_data"]
    assert payload["intensity_col"] == "cell_p95_intensity_ch2"
    assert payload["pathogen_chan"] == 2
    assert payload["log_transform"] is False
    assert payload["intensities_inf"].size == 15
    assert payload["intensities_uninf"].size == 15
    assert payload["bin_edges"].size == 65  # default 64 bins
    assert 200.0 < payload["thr_val"] < 9000.0

    assert "Automatic intensity threshold" in capsys.readouterr().out


def test_infection_qc_histogram_no_intensity_column(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 7, str(tmp_path))

    assert col == "infected"
    assert out is df
    assert "adjusted_infected" not in out.columns
    assert settings["infection_hist_data"] is None
    assert settings["infection_intensity_qc_panel_path"] is None
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    assert "None of" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_infection_qc_histogram_falls_back_to_mean_column(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(col="mean")
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert col == "adjusted_infected"
    assert settings["infection_hist_data"]["intensity_col"] == "cell_mean_intensity_ch2"


def test_infection_qc_histogram_prefers_p95_over_mean(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["cell_mean_intensity_ch2"] = 1.0  # useless column, must be ignored
    settings = {"infection_intensity_qc_graphs": False}
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))
    assert settings["infection_hist_data"]["intensity_col"] == "cell_p95_intensity_ch2"


def test_infection_qc_histogram_too_few_cells_skips(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=8)
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert col == "infected"
    assert "adjusted_infected" not in out.columns
    assert settings["infection_hist_data"] is None
    assert settings["infection_intensity_qc_panel_path"] is None
    assert "Too few cells" in capsys.readouterr().out


def test_infection_qc_histogram_constant_intensity_skips(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=40)
    df["cell_p95_intensity_ch2"] = 500.0  # nunique == 1
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert col == "infected"
    assert settings["infection_hist_data"] is None
    assert "Too few cells or no intensity variation" in capsys.readouterr().out


def test_infection_qc_histogram_drops_stale_adjusted_columns(tmp_path):
    """Re-running on a DB-loaded frame must not produce _x / _y merge suffixes."""
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["adjusted_infected"] = False
    df["adjusted_infected_x"] = 1
    df["adjusted_infected_y"] = 2
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert [c for c in out.columns if c.startswith("adjusted_infected")] == [
        "adjusted_infected"
    ]
    assert _per_cell_positive(out) == 15


def test_infection_qc_histogram_percentile_fallback(tmp_path, capsys):
    """No bin reaches the infected fraction -> percentile fallback threshold."""
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["infected"] = False  # nothing is infected -> frac_inf is all zero
    settings = {
        "infection_intensity_qc_graphs": False,
        "infection_hist_percentile": 10.0,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    thr = settings["infection_hist_data"]["thr_val"]
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean()
    assert thr == pytest.approx(float(np.nanpercentile(raw.to_numpy(), 10.0)))
    assert _per_cell_positive(out) == int((raw >= thr).sum())
    assert settings["infection_hist_data"]["intensities_inf"].size == 0
    assert "Could not find bin with infected" in capsys.readouterr().out


def test_infection_qc_histogram_clamps_bins_fraction_and_percentile(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    settings = {
        "infection_intensity_n_bins": 4,          # clamped up to 10
        "infection_intensity_frac_infected": 0.1,  # clamped up to 0.5
        "infection_hist_percentile": -20.0,        # clamped up to 0.0
        "infection_intensity_qc_graphs": False,
    }
    _infection_qc_histogram(_qc_df(), settings, "infected", 2, str(tmp_path))
    assert settings["infection_hist_data"]["bin_edges"].size == 11

    settings2 = {
        "infection_intensity_n_bins": 5000,        # clamped down to 256
        "infection_intensity_frac_infected": 5.0,  # clamped down to 0.95
        "infection_hist_percentile": 500.0,        # clamped down to 100.0
        "infection_intensity_qc_graphs": False,
    }
    _infection_qc_histogram(_qc_df(), settings2, "infected", 2, str(tmp_path))
    assert settings2["infection_hist_data"]["bin_edges"].size == 257


def test_infection_qc_histogram_remove_mode_drops_conflicting_cells(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(seed=3)
    # Two cells whose mask label contradicts their intensity.
    df.loc[df["cellID"] == 1, "infected"] = True    # dim but called infected
    df.loc[df["cellID"] == 30, "infected"] = False  # bright but called uninfected
    settings = {
        "infection_intensity_mode": "REMOVE",  # case-insensitive
        "infection_intensity_qc_graphs": False,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert col == "adjusted_infected"
    assert set(out["cellID"]) == set(range(1, 31)) - {1, 30}
    assert len(out) == len(df) - 4  # 2 cells x 2 frames
    assert _per_cell_positive(out) == 14
    assert "Removed 2 cells" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())  # graphs disabled


def test_infection_qc_histogram_remove_mode_without_conflicts_keeps_all(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_mode": "remove",
        "infection_intensity_qc_graphs": False,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert len(out) == len(df)
    assert _per_cell_positive(out) == 15


def test_infection_qc_histogram_unknown_mode_falls_back_to_relabel(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    settings = {
        "infection_intensity_mode": "not-a-mode",
        "infection_intensity_qc_graphs": False,
    }
    df = _qc_df()
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))
    assert len(out) == len(df)  # nothing removed
    assert "mode=relabel" in capsys.readouterr().out


def test_infection_qc_histogram_relabels_disagreeing_cells(tmp_path, capsys):
    """Mask says infected but intensity is dim -> the label is flipped."""
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df.loc[df["cellID"] == 2, "infected"] = True  # dim cell mislabelled
    settings = {"infection_intensity_qc_graphs": False}
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert bool(out.loc[out["cellID"] == 2, "adjusted_infected"].iloc[0]) is False
    assert _per_cell_positive(out) == 15
    assert "Adjusted infection labels for 1 cells" in capsys.readouterr().out


def test_infection_qc_histogram_nan_cells_fall_back_to_mask_label(tmp_path):
    """Cells dropped by dropna() get their label back from the mask column."""
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=32)
    df.loc[df["cellID"] == 32, "cell_p95_intensity_ch2"] = np.nan
    df.loc[df["cellID"] == 31, "cell_p95_intensity_ch2"] = np.inf
    settings = {"infection_intensity_qc_graphs": False}
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert out["adjusted_infected"].isna().sum() == 0
    # both were in the infected half of the synthetic split
    assert bool(out.loc[out["cellID"] == 32, "adjusted_infected"].iloc[0]) is True
    assert bool(out.loc[out["cellID"] == 31, "adjusted_infected"].iloc[0]) is True


def test_infection_qc_histogram_graphs_disabled_writes_nothing(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    settings = {"infection_intensity_qc_graphs": False}
    _infection_qc_histogram(_qc_df(), settings, "infected", 2, str(tmp_path))

    assert settings["infection_intensity_qc_panel_path"] is None
    assert settings["infection_hist_data"] is not None  # payload still built
    assert not list(tmp_path.iterdir())
    assert "skipping histogram plot" in capsys.readouterr().out


def test_infection_qc_histogram_png_filename_uses_meta_tag(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    df = pd.concat(
        [_qc_df(plate="plateA", well="C03"), _qc_df(plate="plateB", well="C03", seed=9)],
        ignore_index=True,
    )
    # keep cellIDs unique per plate so the groupby key is well formed
    df.loc[df["plateID"] == "plateB", "cellID"] += 100
    settings = {}
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    assert (tmp_path /
            "infection_intensity_histogram_MULTI_PLATES_C03.png").is_file()


def test_infection_qc_histogram_log_threshold_applied_in_log_space(tmp_path):
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_log": True,
        "infection_intensity_qc_graphs": False,
    }
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    thr_log = settings["infection_hist_data"]["thr_val"]
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean()
    expected = int((raw >= 10.0 ** thr_log).sum())
    assert expected < len(raw)  # the threshold is meant to separate the two modes
    assert _per_cell_positive(out) == expected


def test_infection_qc_histogram_log_payload_is_log_transformed(tmp_path):
    """The plotting payload is in log space when log_transform is on."""
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_log": True,
        "infection_intensity_qc_graphs": True,
    }
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    payload = settings["infection_hist_data"]
    assert payload["log_transform"] is True
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean().to_numpy()
    # The helper offsets by eps = max(min(positive) / 2, 1e-6) before log10.
    eps = max(raw.min() * 0.5, 1e-6)
    both = np.concatenate([payload["intensities_inf"], payload["intensities_uninf"]])
    assert both.max() == pytest.approx(np.log10(raw.max() + eps))
    assert both.min() == pytest.approx(np.log10(raw.min() + eps))
    # log compresses the 2 orders of magnitude between the two populations
    assert both.max() - both.min() < 2.0
    assert (tmp_path / "infection_intensity_histogram_plate1_A01.png").is_file()


def test_infection_qc_histogram_output_feeds_adjusted_panel(tmp_path):
    """End-to-end: the payload the histogram stores renders the panel."""
    from spacr.timelapse import _infection_qc_histogram, _make_adjusted_qc_panel

    df = _qc_df()
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    panel_dir = tmp_path / "panel"
    _make_adjusted_qc_panel(out, col, str(panel_dir), settings, "adjusted")

    png = panel_dir / "infection_qc_panel_adjusted_plate1_A01.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
