"""The invariants that made a set of ``spacr.timelapse`` guards unreachable.

Each block these tests stand for was deleted rather than excluded from
coverage, because no input could reach it: something above it made its
condition impossible. A deleted guard leaves nothing behind to assert on, so
what is asserted here is the invariant that killed it. If one of these stops
holding, the deletion stops being safe and a test says so.

Everything is CPU-only and offline, and every assertion is on a real output:
a returned frame, an emitted payload, the axes of a real figure, or the
feature list the model was actually trained on.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# spacr.timelapse pulls in torch/cellpose and is slow to import lazily.
import spacr.timelapse  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


KEY_COLS = ["plateID", "wellID", "fieldID", "cellID"]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _pca_all_df(n_infected=60, n_uninfected=60, n_frames=2, seed=0,
                pathogen_chan=2, extra=None):
    """Frame-level table whose ``cell_*`` features separate the two classes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_infected + n_uninfected):
        infected = i < n_infected
        if infected:
            base, area, peri, sol = 900.0, 550.0, 100.0, 0.76
        else:
            base, area, peri, sol = 60.0, 400.0, 80.0, 0.86
        base += 200.0 * rng.random()
        for t in range(n_frames):
            row = dict(
                plateID="plate1",
                wellID="A01",
                fieldID=1,
                cellID=i + 1,
                timeid=t,
                infected=bool(infected),
                cell_area=area + rng.normal(0, 2.0),
                cell_perimeter=peri + rng.normal(0, 1.0),
                cell_solidity=sol + rng.normal(0, 0.005),
                **{
                    f"cell_p95_intensity_ch{pathogen_chan}": base + rng.normal(0, 5.0),
                    f"cell_mean_intensity_ch{pathogen_chan}": base * 0.6
                    + rng.normal(0, 5.0),
                },
                cell_mean_intensity_ch0=200.0 + rng.normal(0, 10.0),
            )
            if extra:
                row.update(extra(i, t))
            rows.append(row)
    return pd.DataFrame(rows)


def _run_pca(df, settings=None, infection_col="infected", pathogen_chan=2):
    from spacr.timelapse import _infection_qc_pca_clustering

    settings = dict(settings or {})
    settings.setdefault("infection_intensity_mode", "relabel")
    out, col = _infection_qc_pca_clustering(
        df, settings, infection_col, pathogen_chan, None)
    return out, col, settings


# ===========================================================================
# _infection_qc_pca_clustering: "no finite rows", "all-non-finite column"
# and "fewer than ten cells" could not happen
# ===========================================================================

def test_an_all_infinite_feature_column_never_reaches_the_median_imputer():
    """``tmp.replace`` turns infinities into NaN before the per-cell groupby.

    So in that table ``notna`` and ``isfinite`` are the same test, and the
    degenerate-feature filter -- which needs ten notna values -- throws out any
    column that has no finite value at all. A column of alternating +inf/-inf
    survives a naive ``nunique > 1`` check and is still dropped here, which is
    why the imputer's ``if not m.any(): X[:, j] = 0.0`` branch, the
    ``if not mask_rows.any()`` skip above it and the ``X.shape[0] < 10`` skip
    below it were all unreachable.
    """
    # Constant sign per cell, so the per-cell MEDIAN is +inf or -inf rather
    # than the NaN that mixing the two signs would give: without the replace
    # above the groupby this column would reach the imputer as an entirely
    # non-finite one, which is the state the deleted branch handled.
    def _junk(i, t):
        return {"cell_junk_ch2": np.inf if i % 2 == 0 else -np.inf}

    clean = _pca_all_df()
    dirty = _pca_all_df(extra=_junk)
    assert np.isinf(dirty["cell_junk_ch2"]).all()
    assert dirty["cell_junk_ch2"].nunique() == 2  # survives the nunique filter
    per_cell = dirty.groupby("cellID")["cell_junk_ch2"].median()
    assert np.isinf(per_cell).all()  # and the per-cell median stays infinite

    _out_a, col_a, set_a = _run_pca(clean)
    _out_b, col_b, set_b = _run_pca(dirty)

    assert col_a == col_b == "adjusted_infected"
    coords_a = set_a["infection_pca_data"]["coords"]
    coords_b = set_b["infection_pca_data"]["coords"]
    # The junk column changed nothing: it was dropped before the feature
    # matrix was built, so the embedding is bit-for-bit the clean one.
    assert coords_a.shape == coords_b.shape == (120, 2)
    assert np.allclose(coords_a, coords_b)
    assert np.array_equal(
        set_a["infection_pca_data"]["labels"],
        set_b["infection_pca_data"]["labels"],
    )


def test_rows_with_no_finite_feature_are_dropped_and_ten_still_remain():
    """``mask_rows`` does real work, but it cannot empty the matrix.

    Every kept feature column holds at least ten notna -- hence finite --
    values, and a row carrying one of them has a positive finite count, so
    ``mask_rows`` keeps it. The matrix therefore always has at least ten rows
    and ``mask_rows.any()`` is always true.
    """
    df = _pca_all_df(n_infected=60, n_uninfected=60)
    feature_cols = [c for c in df.columns
                    if c.startswith("cell_") and c != "cellID"]

    # twelve extra cells with no finite value in any feature at all
    blanks = df[df["cellID"] == 1].copy()
    blanks = pd.concat([blanks] * 12, ignore_index=True)
    blanks["cellID"] = np.repeat(np.arange(1000, 1012), len(df[df["cellID"] == 1]))
    blanks[feature_cols] = np.nan
    df = pd.concat([df, blanks], ignore_index=True)

    _out, col, settings = _run_pca(df)

    assert col == "adjusted_infected"
    coords = settings["infection_pca_data"]["coords"]
    # 120 real cells embedded; the 12 all-NaN cells dropped, not the reverse
    n_with_a_finite_feature = int(
        df.groupby("cellID")[feature_cols].median().notna().any(axis=1).sum()
    )
    assert n_with_a_finite_feature == 120
    assert coords.shape == (n_with_a_finite_feature, 2)
    assert coords.shape[0] >= 10
    assert np.isfinite(coords).all()


def test_ten_finite_cells_per_class_is_enough_for_the_ground_truth_split(capsys):
    """The per-class counts are checked once, and 10 passes that check.

    ``inf_vals.size`` *is* ``np.sum(y_int)`` and ``uninf_vals.size`` *is*
    ``np.sum(~y_int)``, so a second ``< 10`` test on the same two numbers can
    never fire once the first has let them through.
    """
    df = _pca_all_df(n_infected=10, n_uninfected=40)
    _out, col, settings = _run_pca(df)

    txt = capsys.readouterr().out
    assert "Not enough cells with finite intensity" not in txt
    assert col == "adjusted_infected"
    assert settings["infection_pca_data"]["coords"].shape == (50, 2)


# ===========================================================================
# _make_intensity_motility_panel: the "empty on both sides" skips and the
# single-column layout could not happen
# ===========================================================================

def _panel_all_df(n_cells=4, n_frames=3, n_channels=2, pathogen_chan=None,
                  all_infected=False):
    rows = []
    for c in range(n_cells):
        inf = True if all_infected else bool(c % 2 == 0)
        for t in range(n_frames):
            row = {"plateID": "plate1", "wellID": "A01", "fieldID": 1,
                   "cellID": c + 1, "infected": inf}
            for ch in range(n_channels):
                row[f"cell_mean_intensity_ch{ch}"] = 100.0 + 10.0 * c + 5.0 * ch + t
            rows.append(row)
    df = pd.DataFrame(rows)
    if pathogen_chan is not None:
        base = df[f"cell_mean_intensity_ch{pathogen_chan}"]
        df[f"cell_p75_intensity_ch{pathogen_chan}"] = base * 1.2
        df[f"pathogen_mean_intensity_ch{pathogen_chan}"] = base * 2.0
        df[f"cytoplasm_mean_intensity_ch{pathogen_chan}"] = base * 0.5
    return df


def _panel_tracks(infected=(True, False, True, False), length=5):
    out = []
    for i, inf in enumerate(infected):
        n = length
        out.append({
            "plateID": "plate1", "wellID": "A01",
            "x_px": 10.0 + np.arange(n, dtype=float) * (i + 1),
            "y_px": 20.0 + np.arange(n, dtype=float) * 0.5 * (i + 1),
            "infected": bool(inf),
        })
    return out


def _panel_track_df(infected=(True, False, True, False)):
    n = len(infected)
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "wellID": ["A01"] * n,
        "infected": list(infected),
        "velocity": np.linspace(0.5, 2.0, n),
    })


@pytest.fixture
def panels(monkeypatch):
    """Capture every ``(fig, axes)`` ``plt.subplots`` hands the panel."""
    captured = []
    orig = plt.subplots

    def _spy(*args, **kwargs):
        fig, axes = orig(*args, **kwargs)
        captured.append((fig, np.atleast_1d(np.asarray(axes)).ravel()))
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _spy)
    return captured


def _make_panel(tmp_path, all_df, n_channels, settings, label_tag,
                infected=(True, False, True, False), tracks=None):
    from spacr.timelapse import _make_intensity_motility_panel

    motility_dir = str(tmp_path / "motility")
    if tracks is None:
        tracks = _panel_tracks(infected)
    _make_intensity_motility_panel(
        all_df=all_df,
        infection_col="infected",
        track_df=_panel_track_df(infected),
        per_well_tracks={"plate1_A01": tracks},
        n_channels=n_channels,
        motility_dir=motility_dir,
        pixels_per_um=2.0,
        seconds_per_frame=30.0,
        vel_unit="um/s",
        settings=settings,
        label_tag=label_tag,
    )
    return motility_dir


def test_the_smallest_panel_is_still_four_axes_wide(tmp_path, panels):
    """``n_cols == 1`` cannot happen, so ``plt.subplots`` never returns a
    bare Axes here.

    ``available_channels`` is non-empty by the guard above the layout maths --
    the well is skipped otherwise -- and the count adds a fixed three motility
    axes on top of it, so the narrowest panel spaCR can draw is four columns.
    """
    _make_panel(
        tmp_path,
        all_df=_panel_all_df(n_channels=1),
        n_channels=1,
        settings={"pathogen_channel": None},
        label_tag="mask_labels",
    )
    assert panels, "the panel never reached plt.subplots"
    _fig, axes = panels[0]
    assert len(axes) == 4
    assert all(hasattr(ax, "get_visible") for ax in axes)


def test_a_well_where_every_cell_is_infected_still_draws_its_violin(tmp_path, panels):
    """One side of the infected/uninfected split can be empty; both cannot.

    ``mask_inf`` and ``~mask_inf`` partition a per-cell table the guard above
    has already established is non-empty, so the violin helper's "nothing to
    draw" skip had no input that reached it.
    """
    _make_panel(
        tmp_path,
        all_df=_panel_all_df(n_channels=2, pathogen_chan=1, all_infected=True),
        n_channels=2,
        settings={"pathogen_channel": 1},
        label_tag="mask_labels",
        infected=(True, True, True, True),
    )
    _fig, axes = panels[0]
    ax = axes[0]
    assert ax.get_visible() is True
    # one violin body for the infected side, none for the empty uninfected one
    assert len(ax.collections) >= 1
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Inf"]


def test_a_well_where_every_cell_is_infected_still_draws_the_qc_histogram(
        tmp_path, panels):
    """Same partition, in the adjusted panel's histogram QC axis."""
    settings = {
        "pathogen_channel": 1,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_n_bins": 8,
        "infection_intensity_threshold": 115.0,
    }
    _make_panel(
        tmp_path,
        all_df=_panel_all_df(n_channels=2, pathogen_chan=1, all_infected=True),
        n_channels=2,
        settings=settings,
        label_tag="adjusted_labels",
        infected=(True, True, True, True),
    )
    _fig, axes = panels[0]
    ax_hist = axes[7]
    assert ax_hist.get_visible() is True
    # both stacks are still drawn -- 8 bins each -- plus the threshold line
    assert len(ax_hist.patches) == 16
    assert len(ax_hist.lines) == 1
    heights = [p.get_height() for p in ax_hist.patches]
    assert sum(heights[:8]) == 0.0        # the empty uninfected side
    assert sum(heights[8:]) == 4.0        # every one of the four cells


def test_every_violin_axis_gets_a_column_the_frame_actually_has(tmp_path,
                                                                panels):
    """The violin helper's "column missing" skip had no caller that could
    reach it.

    Each of its three call sites names a column it has just established is
    present: the per-channel one comes from ``available_channels``, the p75 one
    from ``has_p75_path``, and ``rel_intensity`` is computed on the frame one
    line before the call. So every intensity axis is drawn, never blanked.
    """
    # n_channels claims three, but the frame carries columns for two: the
    # third contributes no axis, because the channel list is built by asking
    # the frame rather than by counting.
    _make_panel(
        tmp_path,
        all_df=_panel_all_df(n_channels=2, pathogen_chan=1),
        n_channels=3,
        settings={"pathogen_channel": 1},
        label_tag="mask_labels",
    )
    _fig, axes = panels[0]
    assert len(axes) == 7        # 2 channels + p75 + ratio + 3 motility axes
    # ch0 mean, ch1 mean, ch1 p75, ch1 pathogen/cytoplasm ratio
    titles = ["Ch 0 mean", "Ch 1 mean", "Ch 1 p75", "Ch 1 pathogen/cytoplasm"]
    for i, title in enumerate(titles):
        assert axes[i].get_title() == title
        assert axes[i].get_visible() is True, title
        assert len(axes[i].collections) >= 1, title


def test_a_well_with_no_tracks_never_reaches_the_track_figure(tmp_path, capsys,
                                                              panels):
    """``_plot_all_tracks`` cannot be handed an empty ``well_tracks``.

    The per-well guard skips the whole well -- no figure at all -- when its
    track list is empty, so the helper's own emptiness check was unreachable.
    """
    # The measurement frame and the per-well track summary both have rows for
    # this well; only the track list is empty, so it is that clause of the
    # per-well guard doing the skipping.
    _make_panel(
        tmp_path,
        all_df=_panel_all_df(n_channels=2, pathogen_chan=1),
        n_channels=2,
        settings={"pathogen_channel": 1},
        label_tag="mask_labels",
        tracks=[],
    )
    out = capsys.readouterr().out
    assert "No data for plate=plate1, well=A01" in out
    assert panels == []                   # the figure was never created


def test_the_infection_feature_selection_only_names_real_columns():
    """``agg_cols`` cannot come out empty once ``feature_cols`` is not.

    The candidates come out of ``schema.model_feature_columns(all_df)``, which
    selects *from* ``all_df.columns``, so re-testing membership afterwards
    could not drop one.
    """
    from spacr import schema
    from spacr.timelapse import _select_infection_feature_columns

    all_df = _pca_all_df(n_infected=30, n_uninfected=30, n_frames=2,
                         pathogen_chan=2)
    all_df = all_df.rename(columns={"timeid": "frame"})

    candidates = schema.model_feature_columns(all_df, allow_unknown=True)
    assert candidates, "nothing was selectable at all"
    assert set(candidates) <= set(all_df.columns)

    picked = _select_infection_feature_columns(all_df, pathogen_chan=2)
    assert picked, "no infection features were selected"
    assert set(picked) <= set(all_df.columns)
    # channel 0 intensities are excluded, channel 2 (the pathogen) is kept
    assert "cell_mean_intensity_ch0" not in picked
    assert "cell_p95_intensity_ch2" in picked
    assert "frame" not in picked


# ===========================================================================
# _infection_qc_histogram: the chosen column is chosen BY membership
# ===========================================================================

def _hist_all_df(pathogen_chan=1, drop=()):
    rng = np.random.default_rng(1)
    rows = []
    for cid in range(1, 41):
        infected = cid <= 20
        base = 900.0 if infected else 100.0
        for t in range(2):
            rows.append({
                "plateID": "plate1", "wellID": "A01", "fieldID": 1,
                "cellID": cid, "frame": t, "infected": infected,
                f"cell_p95_intensity_ch{pathogen_chan}": base + rng.normal(0, 5),
                f"cell_mean_intensity_ch{pathogen_chan}": base * 0.6
                + rng.normal(0, 5),
            })
    return pd.DataFrame(rows).drop(columns=list(drop))


@pytest.mark.parametrize(
    "drop, expected",
    [
        ((), "cell_p95_intensity_ch1"),
        (("cell_p95_intensity_ch1",), "cell_mean_intensity_ch1"),
    ],
    ids=["prefers_p95", "falls_back_to_mean"],
)
def test_the_histogram_qc_only_ever_names_a_column_it_found(tmp_path, drop,
                                                            expected):
    """A second "column not found" skip stood below the candidate loop.

    The loop only assigns ``intensity_col`` from a candidate it has just seen
    in ``all_df.columns``, and the ``is None`` guard covers the one other
    outcome, so re-testing membership afterwards was unreachable.
    """
    from spacr.timelapse import _infection_qc_histogram

    all_df = _hist_all_df(drop=drop)
    settings = {"infection_intensity_mode": "relabel"}
    out, col = _infection_qc_histogram(
        all_df=all_df, settings=settings, infection_col="infected",
        pathogen_chan=1, motility_dir=str(tmp_path))

    assert col == "adjusted_infected"
    assert settings["infection_hist_data"]["intensity_col"] == expected
    assert expected in all_df.columns


def test_the_histogram_qc_skips_only_when_no_candidate_exists(tmp_path, capsys):
    from spacr.timelapse import _infection_qc_histogram

    all_df = _hist_all_df(drop=("cell_p95_intensity_ch1",
                                "cell_mean_intensity_ch1"))
    settings = {}
    out, col = _infection_qc_histogram(
        all_df=all_df, settings=settings, infection_col="infected",
        pathogen_chan=1, motility_dir=str(tmp_path))

    assert col == "infected"
    assert "adjusted_infected" not in out.columns
    assert settings["infection_intensity_qc_panel_path"] is None
    assert "None of" in capsys.readouterr().out


# ===========================================================================
# _infection_qc_xgboost
# ===========================================================================

def _xgb_all_df(cell_specs, n_frames=3, seed=0, pathogen_chan=1):
    rng = np.random.default_rng(seed)
    rows = []
    for cid, (well, infected, intensity) in enumerate(cell_specs, start=1):
        area0 = float(rng.uniform(200.0, 900.0)) + (300.0 if infected else 0.0)
        sol0 = float(rng.uniform(0.70, 0.99))
        for f in range(n_frames):
            rows.append({
                "plateID": "plate1", "wellID": well, "fieldID": "1",
                "cellID": cid, "frame": f, "timeID": f,
                "infected": bool(infected),
                "n_pathogens": 3 if infected else 0,
                f"cell_p95_intensity_ch{pathogen_chan}": float(intensity),
                f"cell_mean_intensity_ch{pathogen_chan}":
                    float(intensity) * 0.6 + float(rng.normal(0, 2.0)),
                "cell_mean_intensity_ch0": float(rng.uniform(100.0, 200.0)),
                "cell_area": area0 + float(rng.normal(0, 5.0)),
                "cell_perimeter": 0.4 * area0 + float(rng.normal(0, 3.0)),
                "cell_solidity": sol0 + float(rng.normal(0, 0.005)),
                "cell_centroid-0": 10.0 + 1.5 * f,
                "cell_centroid-1": 20.0 + 1.0 * f,
            })
    return pd.DataFrame(rows)


def _separable(n_per_class=18, wells=("A01", "A02"), seed=3):
    rng = np.random.default_rng(seed)
    specs = []
    for well in wells:
        for _ in range(n_per_class):
            specs.append((well, True, float(rng.normal(1000.0, 120.0))))
            specs.append((well, False, float(rng.normal(300.0, 120.0))))
    return specs


def _xgb_settings(**over):
    base = {
        "tracked_object": "cell",
        "infection_xgb_n_estimators": 15,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "infection_intensity_mode": "relabel",
    }
    base.update(over)
    return base


def _run_xgb(all_df, settings, motility_dir, infection_col="infected",
             pathogen_chan=1):
    from spacr.timelapse import _infection_qc_xgboost
    return _infection_qc_xgboost(
        all_df=all_df, settings=settings, infection_col=infection_col,
        pathogen_chan=pathogen_chan, motility_dir=str(motility_dir))


def _used_features(stdout):
    """The feature list the model was actually trained on, as printed."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if "features:" in line and "Using" in line:
            n = int(line.split("Using")[1].split()[0])
            names = [f.strip() for f in lines[i + 1].split(",") if f.strip()]
            assert len(names) == n, (line, names)
            return names
    raise AssertionError("the run never printed its feature list:\n" + stdout)


@pytest.mark.parametrize("label_dtype", ["bool", "int64"], ids=["bool", "int"])
def test_the_merge_hands_back_the_infection_column_under_its_own_name(
        tmp_path, label_dtype):
    """A ``<col>_y`` / ``<col>_x`` recovery block and a KeyError stood here.

    ``agg_cols`` excludes the infection column, so the left side of the merge
    cannot carry it and the suffix cannot fire; the right side is
    ``groupby(key_cols)[col].max()``, so it always does. Neither an ``_x`` nor
    a ``_y`` copy of the label can exist in the per-cell table.

    Both label dtypes are driven: an integer 0/1 column is the one a median
    aggregation would happily carry through, so it is the case that would grow
    an ``infected_y`` if the exclusion above ever stopped happening.
    """
    all_df = _xgb_all_df(_separable())
    all_df["infected"] = all_df["infected"].astype(label_dtype)
    out, col = _run_xgb(all_df, _xgb_settings(), tmp_path)

    assert col == "adjusted_infected"
    assert not [c for c in out.columns
                if c.endswith("_x") or c.endswith("_y")], list(out.columns)
    assert "infected" in out.columns
    assert out["adjusted_infected"].notna().all()


def test_a_cell_infected_in_one_frame_is_infected_in_the_per_cell_table(
        tmp_path, capsys):
    """The label reaches the per-cell table from the merge, not the median.

    That is what makes the deleted ``<col>_y`` / ``<col>_x`` recovery
    unreachable: ``agg_cols`` withholds the infection column from the median
    aggregation, so the only copy of it comes from
    ``groupby(key_cols)[col].max()`` on the right of the merge and arrives
    unsuffixed. Any-frame-infected therefore means infected, which a median
    over frames would not give.
    """
    specs = _separable(n_per_class=18, wells=("A01", "A02"))
    all_df = _xgb_all_df(specs, n_frames=3)
    # every infected cell is flagged on exactly one of its three frames
    inf_cells = sorted(all_df.loc[all_df["infected"], "cellID"].unique())
    all_df["infected"] = False
    for cid in inf_cells:
        rows = all_df.index[(all_df["cellID"] == cid) & (all_df["frame"] == 0)]
        all_df.loc[rows, "infected"] = True
    assert all_df.groupby("cellID")["infected"].mean().max() == pytest.approx(1 / 3)

    settings = _xgb_settings()
    out, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert col == "adjusted_infected"
    # A per-frame median would call every one of these cells uninfected and
    # the run would bail to the histogram QC. It trains instead.
    assert "Too few infected or uninfected cells overall" not in txt
    assert "Extreme-intensity candidates: infected=9, uninfected=9" in txt
    assert "infection_prob" in out.columns
    assert settings["infection_xgb_importance"] is not None


def test_the_label_frame_and_time_columns_are_never_model_features(tmp_path,
                                                                   capsys):
    """Skips for ``orig_infection_col``, ``frame`` and ``timeID`` were dead.

    ``agg_cols`` already drops all three before the per-cell table exists, and
    ``schema.model_feature_columns`` would drop them anyway: ``timeID`` is
    provenance, ``frame`` is not a declared measurement, and the infection call
    is cast to bool one block up -- a dtype that selector omits.
    """
    from spacr import schema

    all_df = _xgb_all_df(_separable())
    assert {"frame", "timeID", "infected"}.issubset(all_df.columns)

    _out, _col = _run_xgb(all_df, _xgb_settings(), tmp_path)
    used = _used_features(capsys.readouterr().out)

    assert used, "no features were selected"
    assert "frame" not in used
    assert "timeID" not in used
    assert "infected" not in used

    # the same three, straight through the selector that makes it true
    probe = pd.DataFrame({
        "frame": [0, 1, 2],
        "timeID": [0, 1, 2],
        "infected": pd.Series([True, False, True], dtype=bool),
        "cell_area": [1.0, 2.0, 3.0],
    })
    assert schema.model_feature_columns(probe) == ["cell_area"]


def test_a_single_intensity_value_still_defines_both_quartile_sets(tmp_path,
                                                                   capsys):
    """``hi_inf``/``lo_uninf`` cannot come out empty.

    The percentiles are taken over the finite values of the very columns being
    filtered, so they lie between those values' own min and max: the maximum
    row always satisfies ``>= high_thr_inf`` and the minimum row always
    satisfies ``<= low_thr_uninf``. Flat intensities are the tightest case --
    every value equals the percentile -- and both sets still fill.
    """
    specs = []
    for well in ("A01", "A02"):
        for _ in range(18):
            specs.append((well, True, 1000.0))   # every infected cell identical
            specs.append((well, False, 300.0))   # every uninfected cell identical
    all_df = _xgb_all_df(specs)

    out, col = _run_xgb(all_df, _xgb_settings(), tmp_path)

    txt = capsys.readouterr().out
    assert "Could not define confident high/low quartiles" not in txt
    assert "Extreme-intensity candidates: infected=36, uninfected=36" in txt
    assert col == "adjusted_infected"
    assert "infection_prob" in out.columns


def test_a_well_with_one_cell_per_class_is_trained_on_not_skipped(tmp_path,
                                                                  capsys):
    """The per-well training set can never come out one-sided twice.

    ``n_pos``/``n_neg`` are both non-zero by the guard above, so the small-well
    branch keeps at least one index per class and the balanced branch asks
    ``rng.choice`` for ``min(n_pos, n_neg) >= 1``. A second single-class skip
    below them was unreachable.
    """
    specs = _separable(n_per_class=18, wells=("A01",))
    # A03 contributes exactly one extreme cell of each class -- the tightest
    # input the small-well branch can be handed.
    specs.append(("A03", True, 5000.0))
    specs.append(("A03", False, 1.0))
    # A04 has infected cells only, so its extremes really are one-sided and it
    # is the guard ABOVE -- the one that makes the deleted second test dead --
    # that must catch it.
    specs.append(("A04", True, 5100.0))
    specs.append(("A04", True, 5200.0))
    all_df = _xgb_all_df(specs)

    _out, col = _run_xgb(all_df, _xgb_settings(), tmp_path)

    txt = capsys.readouterr().out
    assert col == "adjusted_infected"
    assert "wells used=2" in txt          # A01 and A03, not A04
    assert "Wells skipped due to single class in extreme set: plate1_A04" in txt


def test_perfectly_correlated_features_still_leave_one_standing(tmp_path,
                                                               capsys):
    """``keep[0]`` is never cleared, so the correlation filter cannot empty
    the feature set and the matrix always has a column.

    The inner loop only ever writes ``keep[j]`` for ``j > i >= 0``, so index 0
    survives whatever the correlation matrix says. Here every feature is a
    linear function of one value, so all of them correlate at 1.0 and the
    filter removes as many as it possibly can.
    """
    specs = _separable()
    all_df = _xgb_all_df(specs)
    # make every cell_* feature an exact multiple of the pathogen intensity
    base = all_df["cell_p95_intensity_ch1"]
    for i, c in enumerate(["cell_mean_intensity_ch1", "cell_area",
                           "cell_perimeter", "cell_solidity"], start=2):
        all_df[c] = base * float(i)

    _out, col = _run_xgb(all_df, _xgb_settings(), tmp_path)

    txt = capsys.readouterr().out
    assert "All features flagged as highly correlated" not in txt
    assert "No usable feature columns after correlation" not in txt
    used = _used_features(txt)
    assert len(used) == 1, used
    assert col == "adjusted_infected"


# ===========================================================================
# spacr.io.preprocess_img_data: the stack builder ran behind an ``if True:``
# ===========================================================================

def _write_tif(path, arr):
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path))


def _tiny_img(seed=0, shape=(16, 16)):
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 200, size=shape, dtype=np.uint16)
    img[4:10, 4:10] = 40000
    return img


ALL_IMG_FORMATS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp',
                   '.nd2', '.czi', '.lif']


@pytest.mark.parametrize("sniffed", ["tif", "none"], ids=["tif_src", "no_images"])
def test_the_stack_builder_runs_whatever_extension_was_sniffed(tmp_path,
                                                               monkeypatch,
                                                               sniffed):
    """The organiser is reached on both extension outcomes, with all nine.

    It used to sit behind an ``if True:`` -- a guard left over from an
    ``img_format is not None`` test that had to go, because with
    ``img_format=None`` nothing built ``stack/`` and the run died two
    functions later on a missing directory. The wrapper is gone; what has to
    stay true is that ``stack/`` is still absent when this runs, the sniffed
    extension is overridden with the full list, and the organiser is called
    exactly once either way.
    """
    import spacr.io as IO
    import spacr.plot as PLOT
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    if sniffed == "tif":
        for i, name in enumerate(("fovA.tif", "fovB.tif")):
            _write_tif(src / name, _tiny_img(seed=i))
    else:
        # nothing with a recognised extension -> img_format sniffs to None
        (src / "notes.txt").write_text("not an image")
        (src / "0").mkdir()

    seen = []

    def _organise(s, regex, batch_size, metadata_type, img_format,
                  timelapse=False, save_original_images=True):
        seen.append({"img_format": img_format,
                     "stack_existed": os.path.isdir(os.path.join(s, "stack"))})
        stack = os.path.join(s, "stack")
        os.makedirs(stack, exist_ok=True)
        for i in range(4):
            np.save(os.path.join(stack, f"plate1_A01_00{i}_T0001.npy"),
                    np.zeros((8, 8, 2), dtype=np.uint16))
        return 2

    concat_calls = []
    monkeypatch.setattr(IO, "_rename_and_organize_image_files", _organise)
    monkeypatch.setattr(IO, "_merge_channels",
                        lambda *a, **k: concat_calls.append("merge"))
    monkeypatch.setattr(IO, "concatenate_and_normalize",
                        lambda **k: concat_calls.append(k["src"]) or "masks")
    monkeypatch.setattr(IO, "_create_movies_from_npy_per_channel",
                        lambda *a, **k: None)
    monkeypatch.setattr(PLOT, "plot_arrays", lambda *a, **k: None)

    settings = {
        "src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "organelle_channel": None, "plot": False,
        "batch_size": 1, "test_mode": False, "timelapse": False,
        "normalize": True,
    }
    _out_settings, out_src = preprocess_img_data(settings)

    assert len(seen) == 1, seen
    assert seen[0]["stack_existed"] is False
    assert seen[0]["img_format"] == ALL_IMG_FORMATS
    assert out_src == str(src)
    assert (src / "stack").is_dir()


# ===========================================================================
# NOT dead: the intensity column really can be missing from the selection
# ===========================================================================

def test_an_infection_call_named_like_the_intensity_column_is_still_a_feature(
        tmp_path, capsys):
    """``feature_cols.append(intensity_col)`` looks unreachable and is not.

    Every ``{object}_*`` name is a declared model feature and matches both the
    object-prefix and pathogen-channel patterns, so the selection loop nearly
    always picks the intensity column up on its own. The exception is the
    caller who points ``infection_col`` at that very column: it is then
    withheld from the median aggregation, arrives from the merge, and is cast
    to bool -- a dtype ``schema.model_feature_columns`` omits. This re-add is
    what stops the model losing its intensity feature in that case, so it was
    kept rather than deleted.
    """
    rng = np.random.default_rng(3)
    rows = []
    cid = 0
    for well in ("A01", "A02"):
        for _ in range(18):
            for infected in (True, False):
                cid += 1
                for f in range(3):
                    rows.append({
                        "plateID": "plate1", "wellID": well, "fieldID": "1",
                        "cellID": cid, "frame": f, "timeID": f,
                        "cell_p95_intensity_ch1": bool(infected),
                        "cell_mean_intensity_ch1": float(
                            rng.normal(1000 if infected else 300, 50)),
                        "cell_area": float(
                            rng.normal(800 if infected else 400, 20)),
                        "cell_solidity": float(rng.uniform(0.7, 0.99)),
                    })
    all_df = pd.DataFrame(rows)
    assert all_df["cell_p95_intensity_ch1"].dtype == bool

    settings = _xgb_settings(infection_xgb_corr_threshold=1.01)  # prune nothing
    _out, col = _run_xgb(all_df, settings, tmp_path,
                         infection_col="cell_p95_intensity_ch1")

    used = _used_features(capsys.readouterr().out)
    assert col == "adjusted_infected"
    assert "cell_p95_intensity_ch1" in used
