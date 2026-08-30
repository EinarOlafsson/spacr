"""Branch coverage for the remaining fall-through paths in ``spacr.timelapse``.

Each test here pins one behaviour that only shows up when a guard in the
module takes the branch nobody had driven yet:

* ``_npz_to_movie``      -- a frame that is not (H, W, C) reaches the writer
  with no channel packing and no RGB->BGR conversion.
* ``_find_optimal_search_range`` -- the "displacement too high" warning is
  keyed on an exact floor, so a fractional increment silences it.
* ``preprocess_pathogen_data`` -- a pathogen table with no ``object_label``.
* ``_make_intensity_motility_panel`` -- an all-uninfected well, and a
  configured pathogen channel whose p75 / pathogen / cytoplasm columns are
  simply not in the frame.
* ``_process_merged_group`` -- a dead (all-NaN) intensity channel, children
  that lie outside every cell, and the empty-helper guards.
* ``_smooth_tracks_and_features`` -- an outlier standing next to a second
  outlier is not interpolated away.
* ``_infection_qc_pca_clustering`` -- a pathogen-channel weight that has no
  surviving pathogen feature to apply itself to.
* ``_apply_infection_intensity_qc`` -- the two "fall back to combined" paths
  when the strategy helper returns no ``adjusted_infected`` column.
* ``_infection_qc_xgboost`` -- a single-feature model (no correlation
  pruning) and a stdout failure while reporting the final counts.
* ``automated_motility_assay`` -- the ambiguous-track filter when the frame
  carries no probability-like column at all.

Everything is CPU-only, offline and headless.

Arcs deliberately left uncovered, with the reason each one cannot be taken
(no test contorts itself to reach these; the proof is the record):

* ``_npz_to_movie`` line 87 falling through to 98: line 84 only lets a 3-D
  frame in when ``shape[2] in [1, 2]`` and line 85 has already taken the 1,
  so ``shape[2] == 2`` always holds.
* ``link_by_iou`` line 388 skipping the IoU write: every label comes from
  ``np.unique(mask)``, so ``mask == label`` has at least one True and the
  union is never 0. The one ``np.unique`` value that matches nothing is NaN,
  and a NaN label raises ``KeyError`` at the ``bool_prev``/``bool_next``
  lookup (numpy hands out a fresh NaN scalar per iteration, and NaN != NaN)
  two lines before the union is computed.
* ``_make_intensity_motility_panel`` lines 2970 / 2974 skipping an XGBoost QC
  axis: ``n_cols`` reserves ``qc_axes_count == 2`` axes for this branch, the
  mask-panel QC axis cannot also be allocated (a tag cannot start with both
  "mask" and "adjusted"), and the per-channel loop consumes at most the
  ``extra_int_plots`` it was allocated -- so ``axis_idx`` is at most
  ``len(axes) - 2`` here.
* ``_smooth_tracks_and_features`` line 4108 -> 4109: ``glitch_frames`` is only
  ever filled from ``range(1, n - 1)``, so ``i_local <= 0 or i_local >= n-1``
  is always false.
* ``_smooth_tracks_and_features`` line 4120 -> 4121: ``s`` is ``g[col]``, so
  ``len(s) == len(idx) == n``, and the block is inside ``if n >= 3``.
* ``_debug_plot_merged_planes`` line 4277 -> 4279: ``norm_intensity`` is built
  by ``for ch_idx in range(n_channels)``, so ``n_channels < 1`` gives it size
  0 and the guard at 4269 has already returned.
* ``_infection_qc_pca_clustering`` lines 4555 / 4628 (the "not installed"
  raises): ``_search_umap`` / ``_search_tsne`` are closures called only from
  ``if embed_method == "umap" and umap is not None`` / the t-SNE equivalent,
  which read the same enclosing local the raise tests.
* ``_apply_infection_intensity_qc`` line 5471 -> 5482: ``parts`` is appended
  to only after ``first_payload_settings`` has been set, so the guard at 5464
  already implies it is not None.
* ``_compute_velocities_and_well_summary`` line 5613 -> 5666: every track
  record dict carries ``straightness``, and ``track_records`` is non-empty
  here, so the column always exists.
* ``_compute_velocities_and_well_summary`` line 5705 -> 5708: the per-well
  groupby uses the plate/well values that survived the earlier
  ``groupby(['plateID','wellID','fieldID','cellID'])``; both drop NaN keys by
  default, so a non-empty ``track_df`` always yields at least one group.
* ``_infection_qc_xgboost`` line 7635 -> 7662: ``intensity_col`` was chosen
  because it is in ``cell_level.columns``, and ``cell_level`` is only ever
  re-bound by row filters afterwards.
* ``_infection_qc_xgboost`` line 7665 -> 7698: ``used_feature_cols`` is
  ``feature_cols``, which the guard at 7245 has already refused when empty
  (and the correlation filter never clears ``keep[0]``).
"""
from __future__ import annotations

import builtins
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cv2 = pytest.importorskip("cv2")

# spacr.timelapse is slow to import (torch/cellpose); do it once at collection.
import spacr.timelapse as tl  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ===========================================================================
# _npz_to_movie -- the frame that is neither 1/2-channel nor (H, W, >=3)
# ===========================================================================

class _FakeWriter:
    """cv2.VideoWriter stand-in that keeps every frame handed to it."""

    instances: list = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.frames = []
        self.released = False
        type(self).instances.append(self)

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.released = True


@pytest.fixture
def fake_writer(monkeypatch):
    _FakeWriter.instances = []
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)
    return _FakeWriter


def test_a_frame_with_a_fourth_axis_reaches_the_writer_unconverted(tmp_path, fake_writer):
    """(H, W, 2, 3) misses every channel branch and is written verbatim.

    The RGB->BGR swap at the writer boundary only fires for a real 3- or
    4-channel image, so the control frame below comes out reversed and the
    4-D one does not.
    """
    stack = np.zeros((8, 10, 2, 3), dtype=np.uint8)
    stack[..., 0, 0] = 11
    stack[..., 1, 0] = 22

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 33  # red

    tl._npz_to_movie([stack, rgb], ["stack.npy", "rgb.npy"], str(tmp_path / "m.avi"))

    (writer,) = fake_writer.instances
    assert writer.released is True
    written_stack, written_rgb = writer.frames

    # not packed into (H, W, 3), and not passed through the BGR swap
    assert written_stack.shape == (8, 10, 2, 3)
    assert written_stack[0, 0].tolist() == [[11, 0, 0], [22, 0, 0]]
    # the control frame proves the swap really happens for (H, W, 3)
    assert written_rgb.shape == (8, 10, 3)
    assert written_rgb[0, 0].tolist() == [0, 0, 33]


# ===========================================================================
# _find_optimal_search_range -- the "too high" warning and its exact floor
# ===========================================================================

class _AlwaysFailingLinker:
    """trackpy stand-in whose ``link`` never succeeds."""

    def __init__(self):
        self.calls = 0

    def link(self, features, search_range, memory):
        self.calls += 1
        raise RuntimeError("no link")


def test_a_fractional_increment_walks_past_the_floor_without_warning(monkeypatch, capsys):
    """The warning compares against ``initial - attempts * increment``.

    Repeated subtraction of an integer increment lands exactly on that floor
    and warns; a fractional increment drifts a hair above it and the same
    exhausted search says nothing.
    """
    linker = _AlwaysFailingLinker()
    monkeypatch.setattr(tl, "tp", linker)

    exact = tl._find_optimal_search_range(
        features=[], initial_search_range=500, increment=10, max_attempts=3
    )
    warned = capsys.readouterr().out
    assert exact == 470
    assert "timelapse_displacement=470 is too high" in warned
    assert linker.calls == 3

    drifted = tl._find_optimal_search_range(
        features=[], initial_search_range=500.0, increment=0.7, max_attempts=3
    )
    quiet = capsys.readouterr().out
    assert drifted > 500.0 - 3 * 0.7          # floating-point drift, not maths
    assert drifted == pytest.approx(497.9)
    assert "is too high" not in quiet


# ===========================================================================
# preprocess_pathogen_data -- no object_label to drop
# ===========================================================================

def _pathogen_frame(with_object_label=True):
    rows = []
    for cell in (1, 2):
        for pathogen in (1, 2):
            row = {
                "plateID": "plate1",
                "rowID": "r1",
                "columnID": "c1",
                "fieldID": "f1",
                "timeID": "t1",
                "cell_id": cell,
                "pathogen_area": 10.0 * pathogen,
            }
            if with_object_label:
                row["object_label"] = 10 * cell + pathogen
            rows.append(row)
    return pd.DataFrame(rows)


def test_pathogen_aggregation_without_an_object_label_still_renames_the_cell_link():
    """``object_label`` is dropped when present and simply absent otherwise.

    Either way the host-cell link becomes the frame's ``object_label``.
    """
    with_label = tl.preprocess_pathogen_data(_pathogen_frame(True))
    without_label = tl.preprocess_pathogen_data(_pathogen_frame(False))

    for out in (with_label, without_label):
        assert len(out) == 2
        assert "cell_id" not in out.columns
        assert sorted(out["object_label"]) == [1, 2]
        assert sorted(out["parasite_count"]) == [2, 2]
        assert out["pathogen_area"].tolist() == [15.0, 15.0]

    # the pathogen's own label never survives, whichever way it arrived
    assert with_label.columns.tolist() == without_label.columns.tolist()


# ===========================================================================
# _make_intensity_motility_panel
# ===========================================================================

PLATE = "plate1"
WELL = "A01"


@pytest.fixture
def panels(monkeypatch):
    """Capture every ``(fig, axes)`` the panel builder creates."""
    captured = []
    orig = plt.subplots

    def _spy(*args, **kwargs):
        fig, axes = orig(*args, **kwargs)
        captured.append((fig, np.atleast_1d(np.asarray(axes)).ravel()))
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _spy)
    return captured


def _panel_all_df(n_cells=4, n_frames=3, n_channels=2, pathogen_chan=None,
                  infected=lambda c: c % 2 == 0):
    rows = []
    for c in range(n_cells):
        for t in range(n_frames):
            row = {
                "plateID": PLATE,
                "wellID": WELL,
                "fieldID": 1,
                "cellID": c + 1,
                "infected": bool(infected(c)),
            }
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


def _panel_track_df(infected=(True, False, True, False)):
    n = len(infected)
    return pd.DataFrame({
        "plateID": [PLATE] * n,
        "wellID": [WELL] * n,
        "infected": list(infected),
        "velocity": np.linspace(0.5, 2.0, n),
    })


def _panel_tracks(infected=(True, False, True, False), length=5):
    out = []
    for i, inf in enumerate(infected):
        x = 10.0 + np.arange(length, dtype=float) * (i + 1)
        y = 20.0 + np.arange(length, dtype=float) * 0.5 * (i + 1)
        out.append({"plateID": PLATE, "wellID": WELL, "x_px": x, "y_px": y,
                    "infected": bool(inf)})
    return out


def _call_panel(tmp_path, name="motility", **over):
    settings = {"pathogen_channel": 1}
    settings.update(over.pop("settings", {}) or {})
    motility_dir = str(tmp_path / name)
    kwargs = dict(
        all_df=_panel_all_df(pathogen_chan=1),
        infection_col="infected",
        track_df=_panel_track_df(),
        per_well_tracks={f"{PLATE}_{WELL}": _panel_tracks()},
        n_channels=2,
        motility_dir=motility_dir,
        pixels_per_um=2.0,
        seconds_per_frame=30.0,
        vel_unit="um/s",
        settings=settings,
        label_tag="mask_labels",
    )
    kwargs.update(over)
    tl._make_intensity_motility_panel(**kwargs)
    return motility_dir


def _xgb_panel_settings(**extra):
    s = {
        "pathogen_channel": 1,
        "infection_intensity_strategy": "xgboost",
        "infection_xgb_importance": {
            "feature_names": ["feat_a"],
            "feature_importances": [1.0],
        },
    }
    s.update(extra)
    return s


def test_an_all_uninfected_well_draws_only_the_uninfected_half(tmp_path, panels):
    """No infected cell -> one violin group and one probability histogram."""
    uninfected = _panel_all_df(pathogen_chan=1, infected=lambda c: False)
    uninfected["infection_prob"] = np.linspace(0.05, 0.45, len(uninfected))
    _call_panel(
        tmp_path,
        name="uninf",
        all_df=uninfected,
        label_tag="adjusted_labels",
        settings=_xgb_panel_settings(),
    )
    _fig, axes = panels[0]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Uninf"]
    assert len(axes[7].patches) == 20          # a single 20-bin histogram
    assert [t.get_text() for t in axes[7].get_legend().get_texts()] == ["Uninfected"]

    # same panel, one infected cell added: both halves appear
    mixed = _panel_all_df(pathogen_chan=1, infected=lambda c: c == 0)
    mixed["infection_prob"] = np.linspace(0.05, 0.95, len(mixed))
    _call_panel(
        tmp_path,
        name="mixed",
        all_df=mixed,
        label_tag="adjusted_labels",
        settings=_xgb_panel_settings(),
    )
    _fig2, axes2 = panels[1]
    assert [t.get_text() for t in axes2[0].get_xticklabels()] == ["Inf", "Uninf"]
    assert len(axes2[7].patches) == 40
    assert [t.get_text() for t in axes2[7].get_legend().get_texts()] == [
        "Uninfected", "Infected",
    ]


def test_a_pathogen_channel_without_its_columns_adds_no_extra_axes(tmp_path, panels):
    """``pathogen_channel`` is set but p75 / pathogen / cytoplasm are absent.

    Neither extra intensity axis is allocated and neither is drawn, so the
    panel is exactly one axis per channel plus the three motility axes.
    """
    out = _call_panel(
        tmp_path,
        name="bare",
        all_df=_panel_all_df(pathogen_chan=None),   # no p75 / pathogen / cytoplasm
        settings={"pathogen_channel": 1},
    )
    assert sorted(os.listdir(out)) == ["plate1_A01.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 5                          # 2 channels + 3 motility axes
    assert [ax.get_title() for ax in axes[:2]] == ["Ch 0 mean", "Ch 1 mean"]
    assert axes[2].get_xlabel() == "x (µm)"        # the all-tracks plot, not p75

    # with the columns present the same settings produce the two extra axes
    _call_panel(tmp_path, name="full", settings={"pathogen_channel": 1})
    _fig2, axes2 = panels[1]
    assert len(axes2) == 7
    assert [ax.get_title() for ax in axes2[:4]] == [
        "Ch 0 mean", "Ch 1 mean", "Ch 1 p75", "Ch 1 pathogen/cytoplasm",
    ]


# ===========================================================================
# _process_merged_group
# ===========================================================================

GH, GW = 40, 48
CELL_AREA = 16 * 16


def _group_planes(t, *, dead_channel, children_outside):
    """[ch0, ch1, cell, nucleus, pathogen] planes for frame ``t``."""
    shift = 2 * t

    cell = np.zeros((GH, GW), dtype=np.float32)
    cell[4:20, 4 + shift:20 + shift] = 1

    nucleus = np.zeros((GH, GW), dtype=np.float32)
    pathogen = np.zeros((GH, GW), dtype=np.float32)
    if children_outside:
        # far from every cell pixel: real objects, zero overlap
        nucleus[30:36, 30:36] = 1
        pathogen[30:33, 40:43] = 1
    else:
        nucleus[8:14, 8 + shift:14 + shift] = 1
        pathogen[16:19, 6 + shift:9 + shift] = 1

    yy, xx = np.mgrid[:GH, :GW]
    ch0 = (0.5 * yy + 0.25 * xx + 3.0 * t).astype(np.float32) + 100.0 * (cell > 0)
    if dead_channel:
        ch1 = np.full((GH, GW), np.nan, dtype=np.float32)
    else:
        ch1 = ch0 * 0.5

    return [ch0.astype(np.float32), ch1, cell, nucleus, pathogen]


def _write_group(tmp_path, name, *, n_frames=3, dead_channel=False,
                 children_outside=False):
    src = tmp_path / name
    merged = src / "merged"
    merged.mkdir(parents=True)
    basenames = []
    for t in range(n_frames):
        arr = np.stack(_group_planes(
            t, dead_channel=dead_channel, children_outside=children_outside,
        )).astype(np.float32)
        bn = f"plate1_A01_1_{t}.npy"
        np.save(merged / bn, arr)
        basenames.append(bn)
    return str(src), basenames


def _run_group(src, basenames):
    # (src, file_basenames, n_channels, cell_chan, nucleus_chan, pathogen_chan)
    return tl._process_merged_group((src, basenames, 2, 0, 0, 1))


def test_a_dead_intensity_channel_contributes_no_percentile_columns(tmp_path):
    """An all-NaN channel yields no percentiles for any compartment.

    The finite channel in the same stack still does, for cell, nucleus,
    pathogen and cytoplasm alike.
    """
    src, basenames = _write_group(tmp_path, "dead", dead_channel=True)
    df = _run_group(src, basenames)

    assert len(df) == 3                                   # one cell x 3 frames
    for prefix in ("cell", "nucleus", "pathogen", "cytoplasm"):
        assert f"{prefix}_p75_intensity_ch0" in df.columns
        assert f"{prefix}_p95_intensity_ch1" not in df.columns
        assert f"{prefix}_p75_intensity_ch1" not in df.columns
    # the dead channel still reaches the mean-intensity table, as NaN
    assert df["cell_mean_intensity_ch1"].isna().all()
    assert df["cell_mean_intensity_ch0"].notna().all()


def test_children_outside_every_cell_produce_no_child_summaries(tmp_path):
    """Nucleus/pathogen objects that overlap no cell are summarised away.

    The cytoplasm, which is carved out of the cell mask itself, always
    overlaps and keeps its summary.
    """
    src, basenames = _write_group(tmp_path, "outside", children_outside=True)
    df = _run_group(src, basenames)

    assert len(df) == 3
    assert "n_nuclei" not in df.columns
    assert "n_pathogens" not in df.columns
    assert "n_cytoplasm" in df.columns
    assert (df["n_cytoplasm"] == 1).all()
    # cytoplasm == the whole cell, because neither child was subtracted from it
    assert np.allclose(df["cytoplasm_area"], CELL_AREA)

    # the same masks placed inside the cell do summarise
    src2, basenames2 = _write_group(tmp_path, "inside")
    inside = _run_group(src2, basenames2)
    assert (inside["n_nuclei"] == 1).all()
    assert (inside["n_pathogens"] == 1).all()


def test_an_empty_mean_intensity_table_leaves_the_geometry_intact(tmp_path, monkeypatch):
    """Failure injection: the per-channel mean helper finds nothing.

    Nothing in the pipeline can produce this -- ``_process_merged_group``
    only gets here with a non-empty cell mask, which always yields a row --
    so the guard is driven at the module seam the worker looks the helper up
    through.
    """
    src, basenames = _write_group(tmp_path, "nomeans")
    baseline = _run_group(src, basenames)
    assert [c for c in baseline.columns if c.startswith("cell_mean_intensity_ch")]

    def _nothing(mask_stack, intensity_stack, channel_index):
        return pd.DataFrame(
            columns=["frame", "track_id", f"cell_mean_intensity_ch{channel_index}"]
        )

    monkeypatch.setattr(tl, "_compute_cell_mean_intensity_per_channel", _nothing)
    df = _run_group(src, basenames)

    assert len(df) == len(baseline)
    assert not [c for c in df.columns if c.startswith("cell_mean_intensity_ch")]
    assert (df["cell_area"] == CELL_AREA).all()       # geometry is untouched
    assert (df["n_nuclei"] == 1).all()


def test_an_empty_cytoplasm_overlap_drops_only_the_cytoplasm_summary(tmp_path, monkeypatch):
    """Failure injection: the cell/cytoplasm overlap table comes back empty.

    The cytoplasm mask is the cell mask minus its children, so it can never
    really miss the cell; the nucleus and pathogen summaries in the same run
    show the guard is specific to the compartment whose overlaps vanished.
    """
    src, basenames = _write_group(tmp_path, "nocyto")
    real = tl._compute_parent_child_overlaps

    def _empty_for_cytoplasm(parent_masks, child_masks, parent_label_col,
                             child_label_col):
        if child_label_col == "cytoplasm_label":
            return pd.DataFrame(columns=["frame", parent_label_col, child_label_col])
        return real(
            parent_masks=parent_masks,
            child_masks=child_masks,
            parent_label_col=parent_label_col,
            child_label_col=child_label_col,
        )

    monkeypatch.setattr(tl, "_compute_parent_child_overlaps", _empty_for_cytoplasm)
    df = _run_group(src, basenames)

    assert len(df) == 3
    assert "n_cytoplasm" not in df.columns
    assert "cytoplasm_area" not in df.columns
    assert (df["n_nuclei"] == 1).all()
    assert (df["n_pathogens"] == 1).all()


# ===========================================================================
# _smooth_tracks_and_features -- a z-score outlier with a loud neighbour
# ===========================================================================

def _smooth_frame(cell_id, values):
    n = len(values)
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "wellID": ["A01"] * n,
        "fieldID": ["1"] * n,
        "cellID": [cell_id] * n,
        "frame": list(range(n)),
        "cell_centroid-0": [10.0] * n,
        "cell_centroid-1": [20.0] * n,
        "cell_area": list(values),
    })


def test_an_outlier_beside_a_second_outlier_is_not_interpolated_away():
    """Interpolation needs both neighbours below half the z threshold.

    Two adjacent spikes keep each other, a lone spike is replaced by the
    mean of its neighbours.
    """
    df = pd.concat(
        [
            _smooth_frame(1, [100.0, 100.0, 1000.0, 1000.0, 100.0, 100.0]),
            _smooth_frame(2, [100.0, 100.0, 1000.0, 100.0, 100.0]),
        ],
        ignore_index=True,
    )
    out = tl._smooth_tracks_and_features(df, max_displacement=50.0, zscore_thresh=1.0)

    pair = out[out["cellID"] == 1]["cell_area"].tolist()
    lone = out[out["cellID"] == 2]["cell_area"].tolist()
    assert pair == [100.0, 100.0, 1000.0, 1000.0, 100.0, 100.0]
    assert lone == [100.0, 100.0, 100.0, 100.0, 100.0]


# ===========================================================================
# _infection_qc_pca_clustering -- a pathogen weight with nothing to weight
# ===========================================================================

def _pca_frame(n_each=60, constant_pathogen=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(2 * n_each):
        infected = i < n_each
        if infected:
            area, peri, sol = 550.0, 100.0, 0.76
        else:
            area, peri, sol = 400.0, 80.0, 0.86
        p95 = 500.0 if constant_pathogen else (900.0 if infected else 100.0)
        rows.append(dict(
            plateID="plate1",
            wellID="A01",
            fieldID=1,
            cellID=i + 1,
            timeID=0,
            infected=bool(infected),
            cell_area=area + rng.normal(0, 2.0),
            cell_perimeter=peri + rng.normal(0, 1.0),
            cell_solidity=sol + rng.normal(0, 0.005),
            cell_p95_intensity_ch2=p95,
            cell_mean_intensity_ch2=p95 * 0.6,
        ))
    return pd.DataFrame(rows)


def _pca_coords(df, weight):
    settings = {
        "infection_intensity_strategy": "pca",
        "infection_pca_pathogen_weight": weight,
    }
    tl._infection_qc_pca_clustering(df.copy(), settings, "infected", 2, None)
    payload = settings["infection_pca_data"]
    assert payload["method_label"] == "PCA"
    return payload["coords"]


def test_a_pathogen_weight_does_nothing_when_every_pathogen_feature_is_degenerate():
    """The weight is applied by feature index, and there may be none.

    A constant pathogen-channel column is still a pathogen column, so the
    ``path_cols`` guard passes, but the degeneracy filter has already taken
    it out of the feature matrix and the up-weighting has no column to hit.
    """
    degenerate = _pca_frame(constant_pathogen=True)
    assert _pca_coords(degenerate, 1.0) == pytest.approx(
        _pca_coords(degenerate, 4.0)
    )

    # a pathogen feature that survives the filter really is up-weighted
    usable = _pca_frame(constant_pathogen=False)
    plain = _pca_coords(usable, 1.0)
    weighted = _pca_coords(usable, 4.0)
    assert plain.shape == weighted.shape
    assert not np.allclose(plain, weighted)


# ===========================================================================
# _apply_infection_intensity_qc -- the two "fall back to combined" paths
# ===========================================================================

def _apply_frame(n_cells=8, plate="plate1", well="A01"):
    rows = []
    for c in range(n_cells):
        rows.append({
            "plateID": plate,
            "wellID": well,
            "fieldID": 1,
            "cellID": c + 1,
            "frame": 0,
            "infected": bool(c % 2),
            "cell_p95_intensity_ch2": 100.0 + 50.0 * c,
        })
    return pd.DataFrame(rows)


def _fake_strategy(calls, adjust):
    """Stand-in for the QC strategy helpers, with or without an adjusted column."""

    def _qc(all_df, settings, infection_col, pathogen_chan, motility_dir):
        calls.append({"n_rows": len(all_df), "settings": settings})
        settings["infection_hist_data"] = "hist-payload"
        settings["infection_intensity_qc_panel_type"] = "histogram"
        df = all_df.copy()
        if adjust:
            df["adjusted_infected"] = [1] * len(df)
            return df, "adjusted_infected"
        return df, infection_col

    return _qc


def _patch_strategies(monkeypatch, calls, adjust):
    for attr in ("_infection_qc_histogram", "_infection_qc_xgboost",
                 "_infection_qc_pca_clustering"):
        monkeypatch.setattr(tl, attr, _fake_strategy(calls, adjust))


@pytest.mark.parametrize(
    "scope, drop_col, expected_msg",
    [
        ("bogus_scope", None,
         "Unknown scope='bogus_scope'; using 'combined' behaviour."),
        ("plate", "plateID",
         "missing grouping columns ['plateID']; falling back to combined QC."),
    ],
    ids=["unknown_scope", "missing_group_column"],
)
def test_a_combined_fallback_keeps_the_helper_column_when_nothing_was_adjusted(
    tmp_path, monkeypatch, capsys, scope, drop_col, expected_msg,
):
    """Both fallbacks return the helper's own label column untouched.

    Whether the scope name is unknown or the grouping column is missing, an
    ``adjusted_infected`` column is only reported when the helper made one.
    """
    df = _apply_frame()
    if drop_col:
        df = df.drop(columns=[drop_col])

    calls = []
    _patch_strategies(monkeypatch, calls, adjust=False)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": scope,
    }
    out, col = tl._apply_infection_intensity_qc(
        df, settings, "infected", 2, str(tmp_path / "mot")
    )
    assert col == "infected"
    assert "adjusted_infected" not in out.columns
    assert len(out) == len(df)
    assert settings["infection_hist_data"] == "hist-payload"   # payload propagated
    assert len(calls) == 1 and calls[0]["n_rows"] == len(df)
    assert expected_msg in capsys.readouterr().out

    # the same fallback does switch columns when the helper adjusts labels
    calls2 = []
    _patch_strategies(monkeypatch, calls2, adjust=True)
    settings2 = {
        "infection_intensity_qc": True,
        "infection_intensity_qc_scope": scope,
    }
    out2, col2 = tl._apply_infection_intensity_qc(
        df, settings2, "infected", 2, str(tmp_path / "mot")
    )
    assert col2 == "adjusted_infected"
    assert out2["adjusted_infected"].tolist() == [1] * len(df)


# ===========================================================================
# _infection_qc_xgboost
# ===========================================================================

XGB_CHAN = 1


def _xgb_frame(n_per_class=18, wells=("A01", "A02"), n_frames=2, seed=3,
               correlated_extra=False):
    """Two wells whose only usable feature is the pathogen p95 intensity."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for well in wells:
        for _ in range(n_per_class):
            for infected in (True, False):
                cid += 1
                base = float(rng.normal(1000.0 if infected else 300.0, 60.0))
                for f in range(n_frames):
                    row = {
                        "plateID": "plate1",
                        "wellID": well,
                        "fieldID": "1",
                        "cellID": cid,
                        "frame": f,
                        "infected": bool(infected),
                        f"cell_p95_intensity_ch{XGB_CHAN}": base,
                        # constant -> dropped by the degeneracy filter
                        "cell_area": 500.0,
                        "cell_solidity": 0.9,
                    }
                    if correlated_extra:
                        row["cell_perimeter"] = base * 2.0 + 1.0
                    rows.append(row)
    return pd.DataFrame(rows)


def _xgb_settings(**over):
    base = {
        "tracked_object": "cell",
        "infection_xgb_n_estimators": 15,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "infection_intensity_mode": "relabel",
        "infection_xgb_drop_ambiguous": False,
    }
    base.update(over)
    return base


def test_a_single_feature_model_skips_correlation_pruning(tmp_path, capsys):
    """With one feature column there is no correlation matrix to prune.

    The same data plus a perfectly correlated second feature does go through
    the filter, which is what makes the single-feature path distinguishable.
    """
    settings = _xgb_settings()
    out, col = tl._infection_qc_xgboost(
        all_df=_xgb_frame(),
        settings=settings,
        infection_col="infected",
        pathogen_chan=XGB_CHAN,
        motility_dir=str(tmp_path / "mot"),
    )
    txt = capsys.readouterr().out
    assert col == "adjusted_infected"
    assert out["adjusted_infected"].dtype.kind == "i"
    assert settings["infection_xgb_importance"]["feature_names"] == [
        f"cell_p95_intensity_ch{XGB_CHAN}"
    ]
    assert "Using 1 cell_* features" in txt
    assert "Removing highly correlated features" not in txt

    settings2 = _xgb_settings()
    tl._infection_qc_xgboost(
        all_df=_xgb_frame(correlated_extra=True),
        settings=settings2,
        infection_col="infected",
        pathogen_chan=XGB_CHAN,
        motility_dir=str(tmp_path / "mot2"),
    )
    txt2 = capsys.readouterr().out
    assert "Removing highly correlated features" in txt2
    assert settings2["infection_xgb_importance"]["feature_names"] == [
        f"cell_p95_intensity_ch{XGB_CHAN}"
    ]


def test_a_failing_stdout_does_not_lose_the_relabelled_frame(tmp_path, monkeypatch):
    """The final count report is best-effort.

    A closed pipe under the "Final infection counts" print (BrokenPipeError
    is what ``python … | head`` raises) is swallowed, and the QC still
    finishes: the payloads after it are written and the frame is returned
    with its adjusted labels.
    """
    seen = []
    real_print = builtins.print

    def _flaky_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        seen.append(msg)
        if msg.startswith("[_infection_qc_xgboost] Final infection counts"):
            raise BrokenPipeError("stdout is gone")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", _flaky_print)
    settings = _xgb_settings()
    out, col = tl._infection_qc_xgboost(
        all_df=_xgb_frame(),
        settings=settings,
        infection_col="infected",
        pathogen_chan=XGB_CHAN,
        motility_dir=str(tmp_path / "mot"),
    )
    monkeypatch.undo()

    assert any(m.startswith("[_infection_qc_xgboost] Final infection counts")
               for m in seen)
    # everything after the swallowed failure still ran
    assert any(m.startswith("[_infection_qc_xgboost] Top XGBoost features")
               for m in seen)
    assert settings["infection_xgb_importance"]["feature_names"]
    assert settings["infection_intensity_qc_panel_type"] == "xgboost"
    assert col == "adjusted_infected"
    assert out["adjusted_infected"].notna().all()


# ===========================================================================
# automated_motility_assay -- ambiguous-track filter with no probability column
# ===========================================================================

ASSAY_TABLE = "timelapse_object_measurements"


@pytest.fixture
def panel_calls(monkeypatch):
    """Replace the heavy panel plotter with a recorder."""
    calls = []
    monkeypatch.setattr(tl, "_make_intensity_motility_panel",
                        lambda **kwargs: calls.append(kwargs))
    return calls


def _assay_src(tmp_path, name="assay", n_channels=3, n_files=2):
    src = tmp_path / name
    merged = src / "merged"
    merged.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n_files):
        arr = rng.integers(0, 500, size=(n_channels, 12, 12)).astype(np.uint16)
        np.save(merged / f"plate1_A01_1_{i}.npy", arr)
    return src


def _assay_frame(n_cells=12, n_frames=4, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(1, n_cells + 1):
        infected = cid % 2 == 0
        y0, x0 = float(rng.uniform(10, 200)), float(rng.uniform(10, 200))
        for f in range(n_frames):
            rows.append({
                "plateID": "plate1",
                "wellID": "A01" if cid <= n_cells // 2 else "A02",
                "fieldID": "1",
                "cellID": cid,
                "frame": f,
                "infected": bool(infected),
                "n_pathogens": 3 if infected else 0,
                f"cell_p95_intensity_ch{XGB_CHAN}": 1000.0 if infected else 300.0,
                "cell_area": 500.0 + float(rng.normal(0, 5.0)),
                "cell_centroid-0": y0 + 1.5 * f,
                "cell_centroid-1": x0 + 1.0 * f,
            })
    return pd.DataFrame(rows)


def _write_assay_db(src, df):
    mdir = src / "measurements"
    mdir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(mdir / "measurements.db") as con:
        df.to_sql(ASSAY_TABLE, con, if_exists="replace", index=False)


def _assay_settings(src, **over):
    base = {
        "src": str(src),
        "channels": [0, 1, 2],
        "cell_channel": 2,
        "nucleus_channel": 0,
        "pathogen_channel": XGB_CHAN,
        "n_jobs": 1,
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "xgboost",
        "infection_intensity_qc_scope": "none",   # QC skipped, strategy kept
        "reuse_existing_measurements": True,
    }
    base.update(over)
    return base


def test_a_frame_with_no_probability_column_skips_ambiguous_track_filtering(
    tmp_path, panel_calls, capsys,
):
    """Auto-discovery runs, finds no candidate at all, and says so.

    The QC scope is off, so nothing wrote a probability column; the same
    assay over a frame that *has* one drops the ambiguous tracks instead.
    """
    src = _assay_src(tmp_path, "no_prob")
    df = _assay_frame()
    assert not [c for c in df.columns if "prob" in c.lower()]
    _write_assay_db(src, df)

    out = tl.automated_motility_assay(_assay_settings(src))
    txt = capsys.readouterr().out
    assert "no XGBoost probability/score column was found" in txt
    assert len(out) == len(df)                     # no track was dropped
    assert "adjusted_infected" not in out.columns  # QC scope 'none'
    # only the mask panel is drawn when the labels were never adjusted
    assert [c["label_tag"] for c in panel_calls] == ["mask_xgboost"]

    # the same settings over a frame carrying a probability column do filter
    src2 = _assay_src(tmp_path, "with_prob")
    df2 = _assay_frame()
    df2["infection_prob"] = np.where(df2["cellID"] % 3 == 0, 0.5, 0.95)
    _write_assay_db(src2, df2)

    out2 = tl.automated_motility_assay(_assay_settings(src2))
    txt2 = capsys.readouterr().out
    assert "no XGBoost probability/score column was found" not in txt2
    assert "ambiguous XGBoost tracks (0.25 < proba < 0.75)" in txt2
    assert (out2["infection_prob"] == 0.95).all()
    assert len(out2) < len(df2)
