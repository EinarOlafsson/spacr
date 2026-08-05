"""CPU coverage for the timelapse correlation / QC-plot helpers.

Covers the four private helpers that sit between
``_compute_track_velocities`` and ``summarise_tracks_from_merged``:

    _feature_velocity_correlations   velocity vs per-track median features
    _make_intensity_sanity_plots     per-channel infected/uninfected bar plots
    _make_motility_plots             combined + per-well track plots
    _select_infection_feature_columns  feature-column selection for infection QC

Everything is synthetic, headless (Agg) and offline. Figures are inspected by
intercepting ``plt.close`` so the *content* of each figure (axis labels, axis
limits, bar heights, plotted coordinates, annotation text) can be asserted
rather than merely checking that a PNG appeared on disk.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# figure plumbing
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Guarantee no figure survives a test (rule 6)."""
    import matplotlib.pyplot as plt
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _png_figure_preference(monkeypatch):
    """Say which figure format these tests are asserting.

    Every figure a pipeline keeps now goes through ``spacr.plot.save_figure``,
    which writes the user's preferred format and rewrites the file extension to
    match. Under pytest there is no preference store, so the preference falls
    back to ``spacr.plot.DEFAULT_FIGURE_FORMAT`` -- PDF. The tests below assert
    exact ``.png`` filenames, so they have to state the preference rather than
    inherit whatever the shipped default happens to be.
    """
    import spacr.plot as P
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 200))


def _snapshot(fig):
    """Freeze everything we want to assert about a figure before it is closed."""
    axes = []
    for ax in fig.axes:
        axes.append(
            {
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "title": ax.get_title(),
                "xlim": tuple(float(v) for v in ax.get_xlim()),
                "ylim": tuple(float(v) for v in ax.get_ylim()),
                "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
                "bar_heights": [
                    float(p.get_height())
                    for p in ax.patches
                    if type(p).__name__ == "Rectangle"
                ],
                "n_boxes": sum(
                    1 for p in ax.patches if type(p).__name__ == "FancyBboxPatch"
                ),
                "n_lines": len(ax.lines),
                "line_colors": [ln.get_color() for ln in ax.lines],
                "line_xdata": [
                    np.asarray(ln.get_xdata(), dtype=float) for ln in ax.lines
                ],
                "line_ydata": [
                    np.asarray(ln.get_ydata(), dtype=float) for ln in ax.lines
                ],
                "n_collections": len(ax.collections),
                "texts": [t.get_text() for t in ax.texts],
            }
        )
    return axes


@pytest.fixture
def captured_figs(monkeypatch):
    """Record a snapshot of every figure the code under test closes."""
    import matplotlib.pyplot as plt

    shots = []
    real_close = plt.close

    def _close(fig=None):
        if hasattr(fig, "axes"):
            shots.append(_snapshot(fig))
        if fig is None:
            return real_close()
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _close)
    return shots


# ---------------------------------------------------------------------------
# synthetic frame builders
# ---------------------------------------------------------------------------

def _build_corr_frames(n_cells=12, infected_of=lambda i: i % 2 == 0):
    """all_df (3 frames/cell) + matching track_df with a perfectly linear feature.

    ``cell_area`` has per-cell MEDIAN == 100 + 10*i but a deliberately skewed
    mean for odd cells, so a correlation of exactly 1.0 against
    ``velocity`` == 1 + 0.5*i can only come out of a *median* aggregation.
    """
    rows = []
    trk = []
    for i in range(n_cells):
        base = 100.0 + 10.0 * i
        skew = 0.0 if i % 2 == 0 else 90.0
        areas = [base - 1.0, base, base + 1.0 + skew]  # median == base
        for f, a in enumerate(areas):
            rows.append(
                {
                    "plateID": "p1",
                    "wellID": "A01",
                    "fieldID": "f1",
                    "cellID": i,
                    "frame": f,
                    "cell_area": a,
                    "nucleus_area": 20.0 + ((i * 7) % 5),
                }
            )
        trk.append(
            {
                "plateID": "p1",
                "wellID": "A01",
                "fieldID": "f1",
                "cellID": i,
                "infected": bool(infected_of(i)),
                "v_px_per_frame": 1.0 + 0.5 * i,
                "straightness": 0.1 + 0.01 * i,
                "velocity": 1.0 + 0.5 * i,
                "velocity_unit": "px/frame",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(trk)


def _track(plate, well, field, cell, infected, xs, ys):
    return {
        "plateID": plate,
        "wellID": well,
        "fieldID": field,
        "cellID": cell,
        "infected": bool(infected),
        "x_px": np.asarray(xs, dtype=float),
        "y_px": np.asarray(ys, dtype=float),
        "v_px_per_frame": 1.0,
        "straightness": 0.5,
    }


# ===========================================================================
# _feature_velocity_correlations
# ===========================================================================

def test_feature_velocity_correlations_empty_track_df_is_a_noop(tmp_path):
    from spacr.timelapse import _feature_velocity_correlations

    all_df, _ = _build_corr_frames()
    assert _feature_velocity_correlations(all_df, pd.DataFrame(), str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


def test_feature_velocity_correlations_writes_csv_with_median_aggregation(
    tmp_path, capsys
):
    from spacr.timelapse import _feature_velocity_correlations

    all_df, track_df = _build_corr_frames(n_cells=12)
    _feature_velocity_correlations(all_df, track_df, str(tmp_path))

    out = tmp_path / "velocity_feature_correlations.csv"
    assert out.is_file()
    got = pd.read_csv(out)

    assert set(got.columns) == {
        "feature",
        "pearson_r",
        "n_tracks",
        "group",
        "abs_pearson_r",
    }
    assert set(got["group"]) == {"all", "infected", "uninfected"}

    # 12 tracks total, 6 infected (even cellIDs), 6 uninfected.
    assert set(got.loc[got["group"] == "all", "n_tracks"]) == {12}
    assert set(got.loc[got["group"] == "infected", "n_tracks"]) == {6}
    assert set(got.loc[got["group"] == "uninfected", "n_tracks"]) == {6}

    # cell_area median is exactly linear in velocity -> r == 1.0.
    r_area = got.loc[
        (got["group"] == "all") & (got["feature"] == "cell_area"), "pearson_r"
    ]
    assert len(r_area) == 1
    assert float(r_area.iloc[0]) == pytest.approx(1.0, abs=1e-9)

    # straightness was also linear in cellID -> r == 1.0 as well.
    r_str = got.loc[
        (got["group"] == "all") & (got["feature"] == "straightness"), "pearson_r"
    ]
    assert float(r_str.iloc[0]) == pytest.approx(1.0, abs=1e-9)

    # abs column and the per-group descending sort.
    assert np.allclose(got["abs_pearson_r"], got["pearson_r"].abs(), equal_nan=True)
    for grp, sub in got.groupby("group"):
        vals = sub["abs_pearson_r"].to_numpy()
        assert np.all(np.diff(vals) <= 1e-12), f"group {grp} not sorted descending"

    assert "Saved velocity–feature" in capsys.readouterr().out


def test_feature_velocity_correlations_skips_group_with_too_few_tracks(
    tmp_path, capsys
):
    """All 8 tracks infected -> 'uninfected' subset is empty and is skipped."""
    from spacr.timelapse import _feature_velocity_correlations

    all_df, track_df = _build_corr_frames(n_cells=8, infected_of=lambda i: True)
    _feature_velocity_correlations(all_df, track_df, str(tmp_path))

    got = pd.read_csv(tmp_path / "velocity_feature_correlations.csv")
    assert set(got["group"]) == {"all", "infected"}
    assert set(got["n_tracks"]) == {8}
    assert "Not enough tracks for correlation (uninfected)." in capsys.readouterr().out


def test_feature_velocity_correlations_all_groups_too_small(tmp_path, capsys):
    from spacr.timelapse import _feature_velocity_correlations

    all_df, track_df = _build_corr_frames(n_cells=4)
    assert _feature_velocity_correlations(all_df, track_df, str(tmp_path)) is None

    out = capsys.readouterr().out
    for label in ("all", "infected", "uninfected"):
        assert f"Not enough tracks for correlation ({label})." in out
    assert not (tmp_path / "velocity_feature_correlations.csv").exists()


def test_feature_velocity_correlations_non_finite_velocities_are_dropped(
    tmp_path, capsys
):
    from spacr.timelapse import _feature_velocity_correlations

    all_df, track_df = _build_corr_frames(n_cells=12)
    # Make 4 tracks (2 infected, 2 uninfected) unusable.
    track_df.loc[[0, 1], "velocity"] = np.nan
    track_df.loc[[2, 3], "velocity"] = np.inf

    _feature_velocity_correlations(all_df, track_df, str(tmp_path))
    got = pd.read_csv(tmp_path / "velocity_feature_correlations.csv")

    assert set(got.loc[got["group"] == "all", "n_tracks"]) == {8}
    # cells 0..3 dropped -> 4 infected (4,6,8,10) and 4 uninfected left,
    # both below the 5-track floor.
    assert set(got["group"]) == {"all"}
    out = capsys.readouterr().out
    assert "Not enough tracks for correlation (infected)." in out
    assert "Not enough tracks for correlation (uninfected)." in out


def test_feature_velocity_correlations_no_numeric_columns(tmp_path, capsys):
    from spacr.timelapse import _feature_velocity_correlations

    all_df = pd.DataFrame(
        {
            "plateID": ["p1"] * 6,
            "wellID": ["A01"] * 6,
            "fieldID": ["f1"] * 6,
            "cellID": [0, 0, 1, 1, 2, 2],
            "frame": [0, 1, 0, 1, 0, 1],
            "timeID": [0, 1, 0, 1, 0, 1],
            "note": ["x"] * 6,
        }
    )
    _, track_df = _build_corr_frames(n_cells=3)
    assert _feature_velocity_correlations(all_df, track_df, str(tmp_path)) is None

    assert "No numeric feature columns" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_feature_velocity_correlations_all_candidates_excluded(tmp_path, capsys):
    """Numeric columns exist but every one of them lands in ``exclude_cols``."""
    from spacr.timelapse import _feature_velocity_correlations

    all_df = pd.DataFrame(
        {
            "plateID": ["p1"] * 6,
            "wellID": ["A01"] * 6,
            "fieldID": ["f1"] * 6,
            "cellID": [0, 0, 1, 1, 2, 2],
            "frame": [0, 1, 0, 1, 0, 1],
            # numeric, survives the frame/timeID/cellID strip, but is in
            # exclude_cols -> no candidate features remain after the merge.
            "v_px_per_frame": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # track_df deliberately carries neither v_px_per_frame nor straightness,
    # so the merge introduces no suffixes and no extra numeric candidates.
    track_df = pd.DataFrame(
        {
            "plateID": ["p1"] * 3,
            "wellID": ["A01"] * 3,
            "fieldID": ["f1"] * 3,
            "cellID": [0, 1, 2],
            "infected": [True, False, True],
            "velocity": [1.0, 2.0, 3.0],
        }
    )
    assert _feature_velocity_correlations(all_df, track_df, str(tmp_path)) is None

    assert "No numeric feature columns" in capsys.readouterr().out
    assert not (tmp_path / "velocity_feature_correlations.csv").exists()


def test_feature_velocity_correlations_swallows_write_failure(tmp_path, capsys):
    """A non-existent output directory must be reported, not raised."""
    from spacr.timelapse import _feature_velocity_correlations

    all_df, track_df = _build_corr_frames(n_cells=12)
    missing = tmp_path / "does" / "not" / "exist"
    assert _feature_velocity_correlations(all_df, track_df, str(missing)) is None

    out = capsys.readouterr().out
    assert "Feature–velocity correlation" in out
    assert "analysis failed with error:" in out
    assert not missing.exists()


# ===========================================================================
# _make_intensity_sanity_plots
# ===========================================================================

def _intensity_frames(values_by_cell, infected_by_cell, channels=(0,)):
    """all_df with one row per (cell, frame) and per-channel mean intensities."""
    rows = []
    for cell, vals in values_by_cell.items():
        for frame, v in enumerate(vals):
            row = {
                "plateID": "p1",
                "wellID": "A01",
                "fieldID": "f1",
                "cellID": cell,
                "frame": frame,
                "infected": int(infected_by_cell[cell]),
            }
            for ch in channels:
                row[f"cell_mean_intensity_ch{ch}"] = v + 100.0 * ch
            rows.append(row)
    return pd.DataFrame(rows)


def test_intensity_sanity_plots_empty_df_creates_nothing(tmp_path):
    from spacr.timelapse import _make_intensity_sanity_plots

    out_dir = tmp_path / "motility"
    assert (
        _make_intensity_sanity_plots(pd.DataFrame(), "infected", 2, str(out_dir))
        is None
    )
    assert not out_dir.exists()


def test_intensity_sanity_plots_bar_heights_are_group_means(
    tmp_path, captured_figs, capsys
):
    from spacr.timelapse import _make_intensity_sanity_plots

    vals = {0: [10.0, 20.0], 1: [30.0, 50.0], 2: [1.0, 3.0], 3: [5.0, 9.0]}
    infected = {0: True, 1: True, 2: False, 3: False}
    all_df = _intensity_frames(vals, infected, channels=(0,))

    out_dir = tmp_path / "motility"
    _make_intensity_sanity_plots(all_df, "infected", 1, str(out_dir))

    png = out_dir / "intensity_channel0_infected_vs_uninfected.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # per-cell means: infected {15, 40}, uninfected {2, 7}
    assert len(captured_figs) == 1
    ax = captured_figs[0][0]
    assert ax["bar_heights"] == pytest.approx([27.5, 4.5])
    assert ax["xticklabels"] == ["Infected", "Uninfected"]
    assert ax["ylabel"] == "Mean cell intensity (channel 0)"
    assert ax["title"] == "Intensity vs infection – channel 0"
    assert ax["ylim"][0] == 0.0
    assert "Saved intensity sanity plot" in capsys.readouterr().out


def test_intensity_sanity_plots_skips_missing_channel_columns(tmp_path, captured_figs):
    from spacr.timelapse import _make_intensity_sanity_plots

    vals = {c: [10.0 * (c + 1), 12.0 * (c + 1)] for c in range(4)}
    infected = {0: True, 1: True, 2: False, 3: False}
    all_df = _intensity_frames(vals, infected, channels=(0, 2))

    out_dir = tmp_path / "motility"
    _make_intensity_sanity_plots(all_df, "infected", 3, str(out_dir))

    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "intensity_channel0_infected_vs_uninfected.png",
        "intensity_channel2_infected_vs_uninfected.png",
    ]
    # channel 1 had no column -> only two figures were built.
    assert len(captured_figs) == 2
    # channel 2 values are the channel-0 values shifted by +200.
    h0 = captured_figs[0][0]["bar_heights"]
    h2 = captured_figs[1][0]["bar_heights"]
    assert h2 == pytest.approx([v + 200.0 for v in h0])


def test_intensity_sanity_plots_all_non_finite_channel_is_skipped(tmp_path, capsys):
    from spacr.timelapse import _make_intensity_sanity_plots

    all_df = _intensity_frames(
        {0: [np.nan, np.inf], 1: [-np.inf, np.nan]},
        {0: True, 1: False},
        channels=(0,),
    )
    out_dir = tmp_path / "motility"
    _make_intensity_sanity_plots(all_df, "infected", 1, str(out_dir))

    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []
    assert "No data for intensity channel 0, skipping sanity plot." in (
        capsys.readouterr().out
    )


def test_intensity_sanity_plots_single_infected_cell_gives_nan_stats(
    tmp_path, captured_figs
):
    """One infected cell only: uninfected mean and both stds are NaN."""
    from spacr.timelapse import _make_intensity_sanity_plots

    all_df = _intensity_frames({0: [42.0, 46.0]}, {0: True}, channels=(0,))
    out_dir = tmp_path / "motility"
    _make_intensity_sanity_plots(all_df, "infected", 1, str(out_dir))

    assert (out_dir / "intensity_channel0_infected_vs_uninfected.png").is_file()
    heights = captured_figs[0][0]["bar_heights"]
    assert heights[0] == pytest.approx(44.0)
    assert np.isnan(heights[1])


# ===========================================================================
# _make_motility_plots
# ===========================================================================

def _motility_inputs():
    """Two wells: A01 mixed infected/uninfected, A02 infected-only."""
    per_well = {
        ("p1", "A01"): [
            _track("p1", "A01", "f1", 1, True, [0.0, 4.0, 8.0], [0.0, 4.0, 12.0]),
            _track("p1", "A01", "f1", 2, False, [20.0, 24.0], [20.0, 28.0]),
        ],
        ("p1", "A02"): [
            _track("p1", "A02", "f1", 3, True, [1.0, 5.0], [1.0, 9.0]),
            _track("p1", "A02", "f1", 4, True, [2.0, 10.0], [2.0, 6.0]),
        ],
    }
    track_df = pd.DataFrame(
        {
            "plateID": ["p1"] * 4,
            "wellID": ["A01", "A01", "A02", "A02"],
            "fieldID": ["f1"] * 4,
            "cellID": [1, 2, 3, 4],
            "infected": [True, False, True, True],
            "velocity": [1.0, 3.0, 5.0, 7.0],
        }
    )
    well_summary_df = pd.DataFrame(
        {
            "plateID": ["p1", "p1"],
            "wellID": ["A01", "A02"],
            "mean_velocity_infected": [1.0, 6.0],
            "mean_velocity_uninfected": [3.0, np.nan],
        }
    )
    return track_df, per_well, well_summary_df


def test_motility_plots_empty_track_df(tmp_path, capsys):
    from spacr.timelapse import _make_motility_plots

    out_dir = tmp_path / "motility"
    assert (
        _make_motility_plots(
            pd.DataFrame(), {("p1", "A01"): []}, pd.DataFrame(),
            str(out_dir), None, None, "px/frame", {},
        )
        is None
    )
    assert "motility plots were not generated." in capsys.readouterr().out
    assert not out_dir.exists()


def test_motility_plots_no_per_well_tracks(tmp_path, capsys):
    from spacr.timelapse import _make_motility_plots

    track_df, _, well_summary_df = _motility_inputs()
    out_dir = tmp_path / "motility"
    assert (
        _make_motility_plots(
            track_df, {}, well_summary_df, str(out_dir),
            4.0, 30.0, "µm/min", {},
        )
        is None
    )
    assert "No per-track velocities available" in capsys.readouterr().out
    assert not out_dir.exists()


def test_motility_plots_physical_units_and_axis_limits(
    tmp_path, captured_figs, capsys
):
    from spacr.timelapse import _make_motility_plots

    track_df, per_well, well_summary_df = _motility_inputs()
    settings = {
        "motility_xlim": (0, 100),
        "motility_ylim": (0, 80),
        "motility_origin_xlim": [-10, 10],
        "motility_origin_ylim": [-5, 5],
    }
    out_dir = tmp_path / "motility"
    _make_motility_plots(
        track_df, per_well, well_summary_df, str(out_dir),
        pixels_per_um=4.0, seconds_per_frame=30.0,
        vel_unit="µm/min", settings=settings,
    )

    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "motility_all_tracks.png",
        "motility_p1_A01_all_tracks.png",
        "motility_p1_A01_infected_origin.png",
        "motility_p1_A01_uninfected_origin.png",
        "motility_p1_A02_all_tracks.png",
        "motility_p1_A02_infected_origin.png",
    ]
    # A02 has no uninfected track -> no uninfected-origin plot for it.
    assert not (out_dir / "motility_p1_A02_uninfected_origin.png").exists()
    assert len(captured_figs) == 6

    combined = captured_figs[0][0]
    assert combined["xlabel"] == "x (µm)"
    assert combined["ylabel"] == "y (µm)"
    assert combined["xlim"] == pytest.approx((0.0, 100.0))
    assert combined["ylim"] == pytest.approx((0.0, 80.0))
    assert combined["n_lines"] == 4          # one polyline per track
    assert combined["n_collections"] == 4    # one end-point scatter per track
    assert combined["n_boxes"] == 1
    # pixels are converted to µm with coord_scale = 1/4.
    assert combined["line_xdata"][0] == pytest.approx([0.0, 1.0, 2.0])
    assert combined["line_ydata"][0] == pytest.approx([0.0, 1.0, 3.0])
    assert sorted(combined["line_colors"]) == ["green", "red", "red", "red"]
    # mean velocities: infected (1+5+7)/3 = 4.333..., uninfected 3.0
    assert combined["texts"] == [
        "Infected (4.33 µm/min)",
        "Uninfected (3.00 µm/min)",
        "1 µm = 4.00 px",
        "1 frame = 30 s",
    ]

    a01_all = captured_figs[1][0]
    assert a01_all["texts"][:2] == [
        "Infected (1.00 µm/min)",
        "Uninfected (3.00 µm/min)",
    ]
    assert a01_all["xlim"] == pytest.approx((0.0, 100.0))

    a01_inf = captured_figs[2][0]
    assert a01_inf["xlim"] == pytest.approx((-10.0, 10.0))
    assert a01_inf["ylim"] == pytest.approx((-5.0, 5.0))
    assert a01_inf["n_lines"] == 1
    # re-centred on the first point: (0,4,8)px - 0 -> /4 -> (0,1,2)
    assert a01_inf["line_xdata"][0] == pytest.approx([0.0, 1.0, 2.0])
    assert a01_inf["texts"] == []

    a01_uninf = captured_figs[3][0]
    assert a01_uninf["n_lines"] == 1
    assert a01_uninf["line_colors"] == ["green"]
    # (20,24)px recentred -> (0,4)px -> /4 -> (0,1)
    assert a01_uninf["line_xdata"][0] == pytest.approx([0.0, 1.0])

    a02_all = captured_figs[4][0]
    # well summary has NaN uninfected velocity for A02 -> "n/a"
    assert a02_all["texts"][:2] == ["Infected (6.00 µm/min)", "Uninfected (n/a µm/min)"]

    a02_inf = captured_figs[5][0]
    assert a02_inf["n_lines"] == 2
    assert a02_inf["line_colors"] == ["red", "red"]

    out = capsys.readouterr().out
    assert "Velocity stats (µm/min): all=4.000" in out
    assert "infected=4.333 (n=3)" in out
    assert "uninfected=3.000 (n=1)" in out
    assert "Saved combined motility plot" in out


def test_motility_plots_pixel_units_without_well_summary(
    tmp_path, captured_figs, capsys
):
    """No calibration, no axis limits, no well summary -> px labels and 'n/a'."""
    from spacr.timelapse import _make_motility_plots

    per_well = {
        ("p1", "B02"): [
            _track("p1", "B02", "f1", 7, False, [0.0, 3.0], [0.0, 4.0]),
        ]
    }
    track_df = pd.DataFrame(
        {
            "plateID": ["p1"],
            "wellID": ["B02"],
            "fieldID": ["f1"],
            "cellID": [7],
            "infected": [False],
            "velocity": [2.5],
        }
    )
    out_dir = tmp_path / "motility"
    _make_motility_plots(
        track_df, per_well, pd.DataFrame(), str(out_dir),
        pixels_per_um=None, seconds_per_frame=None,
        vel_unit="px/frame", settings={},
    )

    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "motility_all_tracks.png",
        "motility_p1_B02_all_tracks.png",
        "motility_p1_B02_uninfected_origin.png",
    ]
    # no infected tracks -> no infected-origin plot at all
    assert not (out_dir / "motility_p1_B02_infected_origin.png").exists()
    assert len(captured_figs) == 3

    combined = captured_figs[0][0]
    assert combined["xlabel"] == "x (pixels)"
    assert combined["ylabel"] == "y (pixels)"
    # coord_scale == 1.0: raw pixel coordinates are plotted unchanged
    assert combined["line_xdata"][0] == pytest.approx([0.0, 3.0])
    assert combined["texts"] == [
        "Infected (n/a px/frame)",
        "Uninfected (2.50 px/frame)",
        "1 µm = ? px",
        "1 frame = ? s",
    ]

    per_well_fig = captured_figs[1][0]
    # empty well_summary_df -> summary_row is None -> both means n/a
    assert per_well_fig["texts"][:2] == [
        "Infected (n/a px/frame)",
        "Uninfected (n/a px/frame)",
    ]
    # no axis limits requested -> autoscaled, not the settings values
    assert per_well_fig["xlim"] != (0.0, 100.0)

    origin_fig = captured_figs[2][0]
    assert origin_fig["n_lines"] == 1
    assert origin_fig["line_colors"] == ["green"]
    assert origin_fig["line_xdata"][0] == pytest.approx([0.0, 3.0])
    assert origin_fig["texts"] == []

    out = capsys.readouterr().out
    assert "all=2.500" in out and "infected=nan (n=0)" in out


def test_motility_plots_ignores_malformed_axis_limits(tmp_path, captured_figs):
    """xlim/ylim of the wrong length are ignored rather than raising."""
    from spacr.timelapse import _make_motility_plots

    track_df, per_well, well_summary_df = _motility_inputs()
    out_dir = tmp_path / "motility"
    _make_motility_plots(
        track_df, per_well, well_summary_df, str(out_dir),
        pixels_per_um=2.0, seconds_per_frame=15.5,
        vel_unit="µm/min",
        settings={"motility_xlim": (0, 10, 20), "motility_ylim": []},
    )
    combined = captured_figs[0][0]
    assert combined["xlim"] != pytest.approx((0.0, 10.0))
    # seconds_per_frame formatted with %g
    assert combined["texts"][3] == "1 frame = 15.5 s"
    assert combined["texts"][2] == "1 µm = 2.00 px"
    assert (out_dir / "motility_all_tracks.png").is_file()


# ===========================================================================
# _select_infection_feature_columns
# ===========================================================================

def _feature_frames(n_cells=15, n_frames=2, extra=None):
    rows = []
    for i in range(n_cells):
        for f in range(n_frames):
            row = {
                "plateID": "p1",
                "wellID": "A01",
                "fieldID": "f1",
                "cellID": i,
                "frame": f,
                "timeID": f,
                "cell_area": 100.0 + i,
                "cell_centroid-0": 5.0 * i,
                "cell_centroid-1": 7.0 * i,
                "blob_idx": float(i),
                "cell_mean_intensity_ch0": 3.0 * i,
                "cell_mean_intensity_ch1": 4.0 * i,
                "cell_mean_intensity_ch2": 5.0 * i,
                "cell_mean_intensity_chX": 6.0 * i,
                "constant_feat": 1.0,
                "sparse_feat": float(i) if i < 5 else np.nan,
                "n_pathogens": i % 3,
                "velocity": 0.5 * i,
                "straightness": 0.1 * i,
                "label_text": "abc",
            }
            if extra:
                for k, fn in extra.items():
                    row[k] = fn(i)
            rows.append(row)
    return pd.DataFrame(rows)


def test_select_infection_feature_columns_pathogen_channel_filter():
    from spacr.timelapse import _select_infection_feature_columns

    all_df = _feature_frames()
    got = _select_infection_feature_columns(all_df, pathogen_chan=2)

    # kept: informative non-centroid features + only the pathogen channel's
    # intensity + the un-parseable "chX" intensity column.
    assert got == ["cell_area", "cell_mean_intensity_ch2", "cell_mean_intensity_chX"]
    # explicitly dropped for the documented reasons
    for dropped in (
        "frame", "timeID", "cellID", "n_pathogens", "velocity", "straightness",
        "blob_idx", "cell_centroid-0", "cell_centroid-1",
        "cell_mean_intensity_ch0", "cell_mean_intensity_ch1",
        "constant_feat", "sparse_feat", "label_text",
    ):
        assert dropped not in got


def test_select_infection_feature_columns_none_channel_keeps_all_intensities():
    from spacr.timelapse import _select_infection_feature_columns

    all_df = _feature_frames()
    got = _select_infection_feature_columns(all_df, pathogen_chan=None)

    assert got == [
        "cell_area",
        "cell_mean_intensity_ch0",
        "cell_mean_intensity_ch1",
        "cell_mean_intensity_ch2",
        "cell_mean_intensity_chX",
    ]


def test_select_infection_feature_columns_unparseable_channel_index_is_kept():
    """A channel suffix that is a *digit* but not int()-able keeps the column."""
    from spacr.timelapse import _select_infection_feature_columns

    # '²'.isdigit() is True but int('²') raises ValueError -> except branch.
    weird = "cell_mean_intensity_ch²"
    all_df = _feature_frames(extra={weird: lambda i: 2.0 * i})
    got = _select_infection_feature_columns(all_df, pathogen_chan=2)

    assert weird in got
    assert "cell_mean_intensity_ch0" not in got


def test_select_infection_feature_columns_drops_low_coverage_and_constant():
    from spacr.timelapse import _select_infection_feature_columns

    # exactly 10 finite cells -> kept; 9 -> dropped; jitter below 1e-6 -> dropped
    all_df = _feature_frames(
        n_cells=20,
        extra={
            "ten_finite": lambda i: float(i) if i < 10 else np.nan,
            "nine_finite": lambda i: float(i) if i < 9 else np.nan,
            "near_constant": lambda i: 5.0 + 1e-9 * i,
        },
    )
    got = _select_infection_feature_columns(all_df, pathogen_chan=None)

    assert "ten_finite" in got
    assert "nine_finite" not in got
    assert "near_constant" not in got


def test_select_infection_feature_columns_returns_empty_when_all_excluded():
    from spacr.timelapse import _select_infection_feature_columns

    all_df = pd.DataFrame(
        {
            "plateID": ["p1"] * 4,
            "wellID": ["A01"] * 4,
            "fieldID": ["f1"] * 4,
            "cellID": [0, 0, 1, 1],
            "frame": [0, 1, 0, 1],
            "timeID": [0, 1, 0, 1],
            "n_pathogens": [1, 2, 1, 0],
            "velocity": [0.1, 0.2, 0.3, 0.4],
            "straightness": [0.5, 0.6, 0.7, 0.8],
            "cell_centroid-0": [1.0, 2.0, 3.0, 4.0],
            "run_idx": [0, 1, 2, 3],
            "well_label": ["a", "b", "c", "d"],
        }
    )
    assert _select_infection_feature_columns(all_df, pathogen_chan=1) == []
