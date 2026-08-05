"""CPU coverage for the merged-plane debug plotter and the embedding-based
infection QC in :mod:`spacr.timelapse`.

Covers ``_debug_plot_merged_planes`` (all re-orientation branches, the
``_generate_mask_random_cmap`` ``NameError`` fallback, degenerate masks and
flat intensity channels) and ``_infection_qc_pca_clustering`` (every early
return, the PCA / UMAP / t-SNE embedding branches with and without their
hyper-parameter searches, ``relabel`` vs ``remove`` modes, sub-sampling and
the debug-plot failure path).

UMAP is never really imported: a fake ``umap`` module is injected into
``sys.modules`` so the tests stay fast and offline.  t-SNE and KMeans are
likewise replaced with deterministic stand-ins where a specific clustering
outcome is required.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def png_figure_preference(monkeypatch):
    """Say which figure format the embedding-QC tests are asserting.

    Every figure a pipeline keeps now goes through ``spacr.plot.save_figure``,
    which follows the user's figure-format preference and rewrites the file
    extension to match. Under pytest there is no preference store, so the
    preference falls back to ``spacr.plot.DEFAULT_FIGURE_FORMAT`` -- PDF. The
    ``_infection_qc_pca_clustering`` tests assert an exact
    ``infection_<method>_qc_embedding.png``, so they state the preference
    rather than inherit the shipped default.

    Deliberately NOT autouse: the ``_debug_plot_merged_planes`` tests in this
    same file assert ``.pdf`` names, and they are meant to keep following the
    shipped default.
    """
    import spacr.plot as P
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 200))


def _tl():
    import spacr.timelapse as tl
    return tl


def _spy_subplots(monkeypatch):
    """Record the ``ncols`` every ``plt.subplots`` call asks for."""
    import matplotlib.pyplot as plt

    recorded = []
    real = plt.subplots

    def spy(*args, **kwargs):
        if len(args) >= 2:
            recorded.append(args[1])
        elif "ncols" in kwargs:
            recorded.append(kwargs["ncols"])
        return real(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy)
    return recorded


def _write_merged(tmp_path, name, arr):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    np.save(merged / name, arr)
    # np.save appends .npy when the name has no suffix
    return merged / name


def _stack_planes(planes):
    """(P, H, W) -> (H, W, P) as stored by spacr's merged .npy files."""
    return np.moveaxis(np.asarray(planes), 0, -1)


def _blob_mask(h, w, labels):
    m = np.zeros((h, w), dtype=np.int32)
    step = max(1, h // (len(labels) + 1))
    for i, lab in enumerate(labels):
        r0 = (i + 1) * step - 1
        m[r0:r0 + 2, 2:w - 2] = lab
    return m


# ---------------------------------------------------------------------------
# synthetic per-frame measurement table for the infection QC
# ---------------------------------------------------------------------------

def make_all_df(
    n_infected=50,
    n_uninfected=50,
    n_frames=3,
    n_mislabeled=0,
    seed=0,
    negative_intensity=False,
):
    """Frame-level table with cell_* features that separate cleanly.

    ``n_mislabeled`` cells carry ``infected=True`` but uninfected-looking
    features, which is what makes ``mode='remove'`` actually remove things.
    """
    rng = np.random.default_rng(seed)
    rows = []
    cell_id = 0
    for i in range(n_infected + n_uninfected):
        labelled_infected = i < n_infected
        # the last `n_mislabeled` "infected" cells look uninfected
        looks_infected = labelled_infected and not (
            n_mislabeled and n_infected - n_mislabeled <= i < n_infected
        )
        cell_id += 1
        if looks_infected:
            base = 900.0 + 200.0 * rng.random()
            area = 550.0 + 40.0 * rng.random()
            peri = 100.0 + 8.0 * rng.random()
            sol = 0.76 + 0.05 * rng.random()
        else:
            base = 60.0 + 30.0 * rng.random()
            area = 400.0 + 40.0 * rng.random()
            peri = 80.0 + 8.0 * rng.random()
            sol = 0.86 + 0.05 * rng.random()
        if negative_intensity and i == 0:
            base = -25.0
        for t in range(n_frames):
            rows.append(
                dict(
                    plateID="plate1",
                    wellID="A01",
                    fieldID=1,
                    cellID=cell_id,
                    timeid=t,
                    infected=bool(labelled_infected),
                    cell_area=area + rng.normal(0, 2.0),
                    cell_perimeter=peri + rng.normal(0, 1.0),
                    cell_solidity=sol + rng.normal(0, 0.005),
                    cell_p95_intensity_ch2=base + rng.normal(0, 5.0),
                    cell_mean_intensity_ch2=base * 0.6 + rng.normal(0, 5.0),
                    cell_mean_intensity_ch0=200.0 + rng.normal(0, 10.0),
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fake embedders
# ---------------------------------------------------------------------------

def _install_fake_umap(monkeypatch, fail_when=None, scale_by="n_neighbors"):
    """Install a deterministic fake at spaCR's lazy UMAP boundary."""
    from spacr import utils

    mod = types.ModuleType("umap")
    calls = []

    class UMAP:
        def __init__(self, n_components=2, random_state=0, n_neighbors=15,
                     min_dist=0.1, **kw):
            self.n_neighbors = int(n_neighbors)
            self.min_dist = float(min_dist)

        def fit_transform(self, X):
            calls.append((self.n_neighbors, self.min_dist))
            if fail_when is not None and fail_when(self.n_neighbors, self.min_dist):
                raise RuntimeError("synthetic UMAP failure")
            scale = self.n_neighbors if scale_by == "n_neighbors" else 1.0
            return np.asarray(X[:, :2], dtype=float) * float(scale)

    mod.UMAP = UMAP
    monkeypatch.setattr(utils, "umap", mod)
    return calls


def _install_fake_tsne(monkeypatch, always_fail=False):
    import sklearn.manifold

    calls = []

    class FakeTSNE:
        def __init__(self, n_components=2, random_state=0, init="pca",
                     learning_rate="auto", perplexity=30.0, **kw):
            self.perplexity = float(perplexity)
            self.learning_rate = learning_rate

        def fit_transform(self, X):
            calls.append((self.perplexity, self.learning_rate))
            if always_fail:
                raise RuntimeError("synthetic t-SNE failure")
            return np.asarray(X[:, :2], dtype=float) * (self.perplexity / 15.0)

    monkeypatch.setattr(sklearn.manifold, "TSNE", FakeTSNE)
    return calls


def _install_const_kmeans(monkeypatch, value):
    """KMeans stand-in that puts every sample in the same cluster."""
    import sklearn.cluster

    class ConstKMeans:
        def __init__(self, n_clusters=2, random_state=0, n_init="auto", **kw):
            self.n_clusters = n_clusters

        def fit_predict(self, coords):
            return np.full(coords.shape[0], value, dtype=int)

    monkeypatch.setattr(sklearn.cluster, "KMeans", ConstKMeans)


# ===========================================================================
# _debug_plot_merged_planes
# ===========================================================================

def test_debug_plot_missing_file_returns_early(tmp_path, capsys):
    from spacr.timelapse import _debug_plot_merged_planes

    out = tmp_path / "out"
    out.mkdir()
    assert _debug_plot_merged_planes(str(tmp_path), "nope.npy", 2, 0, 1, str(out)) is None
    assert "File not found" in capsys.readouterr().out
    assert list(out.iterdir()) == []


def test_debug_plot_hwp_three_channels_two_masks(tmp_path, monkeypatch, capsys):
    """(Y, X, planes) input: 3 intensity channels + 2 masks + merged panel."""
    from spacr.timelapse import _debug_plot_merged_planes

    rng = np.random.default_rng(3)
    h = w = 24
    chans = [rng.integers(0, 500, size=(h, w)).astype(np.uint16) for _ in range(3)]
    m1 = _blob_mask(h, w, [1, 2, 3])
    m2 = _blob_mask(h, w, [1, 2])
    arr = _stack_planes(chans + [m1, m2])
    _write_merged(tmp_path, "sample_A.npy", arr)

    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "sample_A.npy", 3, 0, 2, str(out))

    pdf = out / "merged_planes_sample_A.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
    # 3 channels + 2 masks + 1 merged overlay
    assert ncols == [6]
    txt = capsys.readouterr().out
    assert "original_shape=(24, 24, 5)" in txt
    assert "reoriented_shape=(5, 24, 24)" in txt
    assert "Saved merged plane debug figure" in txt


def test_debug_plot_planes_first_layout_no_masks(tmp_path, monkeypatch):
    """arr.shape[0] == n_channels keeps the array as-is (plane-first)."""
    from spacr.timelapse import _debug_plot_merged_planes

    rng = np.random.default_rng(4)
    arr = rng.integers(0, 900, size=(2, 16, 16)).astype(np.uint16)
    _write_merged(tmp_path, "pf.npy", arr)
    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "pf.npy", 2, 0, 1, str(out))
    assert (out / "merged_planes_pf.pdf").is_file()
    assert ncols == [2]  # no mask planes -> no overlay panel


def test_debug_plot_flat_channel_and_empty_masks(tmp_path, monkeypatch, capsys):
    """A constant channel normalises to zeros; all-zero masks kill the overlay."""
    from spacr.timelapse import _debug_plot_merged_planes

    h = w = 20
    flat = np.full((h, w), 7, dtype=np.uint16)
    grad = np.tile(np.arange(w, dtype=np.uint16), (h, 1))
    zero_mask_a = np.zeros((h, w), dtype=np.int32)
    zero_mask_b = np.zeros((h, w), dtype=np.int32)
    arr = _stack_planes([flat, grad, zero_mask_a, zero_mask_b])
    _write_merged(tmp_path, "flat.npy", arr)

    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "flat.npy", 2, 0, 1, str(out))

    assert (out / "merged_planes_flat.pdf").is_file()
    # 2 channels + 2 masks, but combined_mask is None -> no extra column
    assert ncols == [4]
    assert "Saved merged plane debug figure" in capsys.readouterr().out


def test_debug_plot_4d_timepoint_moveaxis(tmp_path, monkeypatch):
    """4-D (T, Y, X, planes) takes the first timepoint."""
    from spacr.timelapse import _debug_plot_merged_planes

    rng = np.random.default_rng(5)
    arr = rng.integers(0, 400, size=(2, 12, 12, 3)).astype(np.uint16)
    _write_merged(tmp_path, "t4.npy", arr)
    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "t4.npy", 2, 0, 1, str(out))
    assert (out / "merged_planes_t4.pdf").is_file()
    # 2 intensity channels + 1 mask plane; mask plane has non-zero labels
    assert ncols[0] in (3, 4)


def test_debug_plot_4d_collapses_time_into_planes(tmp_path, monkeypatch, capsys):
    """last axis smaller than n_channels -> reshape(-1, H, W) fallback."""
    from spacr.timelapse import _debug_plot_merged_planes

    rng = np.random.default_rng(6)
    arr = rng.integers(0, 300, size=(2, 3, 10, 2)).astype(np.uint16)
    _write_merged(tmp_path, "collapse.npy", arr)
    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "collapse.npy", 5, 0, 1, str(out))
    assert "reoriented_shape=(6, 10, 2)" in capsys.readouterr().out
    assert (out / "merged_planes_collapse.pdf").is_file()
    assert ncols[0] >= 5


def test_debug_plot_2d_array_is_skipped(tmp_path, capsys):
    from spacr.timelapse import _debug_plot_merged_planes

    _write_merged(tmp_path, "flat2d.npy", np.zeros((8, 8), dtype=np.uint16))
    out = tmp_path / "out"
    out.mkdir()
    assert _debug_plot_merged_planes(str(tmp_path), "flat2d.npy", 2, 0, 1, str(out)) is None
    assert "Expected 3D array after reorientation" in capsys.readouterr().out
    assert list(out.iterdir()) == []


def test_debug_plot_clamps_n_channels_to_available_planes(tmp_path, monkeypatch):
    from spacr.timelapse import _debug_plot_merged_planes

    rng = np.random.default_rng(7)
    arr = _stack_planes([rng.integers(0, 100, size=(9, 9)).astype(np.uint16),
                         rng.integers(0, 100, size=(9, 9)).astype(np.uint16)])
    _write_merged(tmp_path, "clamp.npy", arr)
    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "clamp.npy", 5, 0, 1, str(out))
    # n_channels clamped from 5 down to the 2 available planes
    assert ncols == [2]
    assert (out / "merged_planes_clamp.pdf").is_file()


def test_debug_plot_zero_channels_returns_before_plotting(tmp_path, capsys):
    from spacr.timelapse import _debug_plot_merged_planes

    arr = _stack_planes([np.ones((6, 6), dtype=np.uint16)])
    _write_merged(tmp_path, "zeroch.npy", arr)
    out = tmp_path / "out"
    out.mkdir()
    assert _debug_plot_merged_planes(str(tmp_path), "zeroch.npy", 0, 0, 1, str(out)) is None
    assert "No intensity channels to plot" in capsys.readouterr().out
    assert list(out.iterdir()) == []


def test_debug_plot_falls_back_when_cmap_helper_missing(tmp_path, monkeypatch):
    """Deleting _generate_mask_random_cmap exercises the NameError fallback."""
    from spacr.timelapse import _debug_plot_merged_planes
    tl = _tl()

    h = w = 18
    chan = np.tile(np.arange(w, dtype=np.uint16), (h, 1))
    m1 = _blob_mask(h, w, [1, 2])
    m2 = _blob_mask(h, w, [1])
    arr = _stack_planes([chan, m1, m2])
    _write_merged(tmp_path, "nofallback.npy", arr)

    monkeypatch.delattr(tl, "_generate_mask_random_cmap")
    out = tmp_path / "out"
    out.mkdir()
    ncols = _spy_subplots(monkeypatch)
    _debug_plot_merged_planes(str(tmp_path), "nofallback.npy", 1, 0, 1, str(out))

    pdf = out / "merged_planes_nofallback.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
    # 1 channel + 2 masks + merged overlay
    assert ncols == [4]


# ===========================================================================
# _infection_qc_pca_clustering - early returns
# ===========================================================================

def test_qc_empty_df_short_circuits(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = pd.DataFrame()
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert out is df and col == "infected"
    assert "all_df is empty" in capsys.readouterr().out


def test_qc_missing_infection_col(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=2, n_uninfected=2, n_frames=1).drop(columns=["infected"])
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert out is df and col == "infected"
    assert "missing" in capsys.readouterr().out


def test_qc_unsupported_mode(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=2, n_uninfected=2, n_frames=1)
    settings = {"infection_intensity_mode": "DELETE_EVERYTHING"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert out is df and col == "infected"
    assert "Unsupported mode" in capsys.readouterr().out


def test_qc_missing_key_column_raises_keyerror():
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=2, n_uninfected=2, n_frames=1).drop(columns=["fieldID"])
    with pytest.raises(KeyError, match="fieldID"):
        _infection_qc_pca_clustering(df, {}, "infected", 2, None)


def test_qc_no_numeric_cell_features(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = pd.DataFrame({
        "plateID": ["p"] * 4,
        "wellID": ["A01"] * 4,
        "fieldID": [1] * 4,
        "cellID": [1, 2, 3, 4],
        "infected": [True, False, True, False],
    })
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert out is df and col == "infected"
    assert "No numeric cell_* features" in capsys.readouterr().out


def test_qc_no_pathogen_channel_given(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=5, n_uninfected=5, n_frames=1)
    out, col = _infection_qc_pca_clustering(df, {}, "infected", None, None)
    assert col == "infected"
    assert "adjusted_infected" not in out.columns
    assert "No pathogen-channel cell_* intensity column" in capsys.readouterr().out


def test_qc_pathogen_channel_without_matching_column(capsys):
    """The candidate loop runs but nothing matches channel 99."""
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=5, n_uninfected=5, n_frames=1)
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 99, None)
    assert col == "infected"
    assert "No pathogen-channel cell_* intensity column" in capsys.readouterr().out


def test_qc_all_features_degenerate(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    n = 30
    df = pd.DataFrame({
        "plateID": ["p"] * n,
        "wellID": ["A01"] * n,
        "fieldID": [1] * n,
        "cellID": list(range(1, n + 1)),
        "infected": [True, False] * (n // 2),
        "cell_area": [100.0] * n,                 # constant -> dropped
        "cell_p95_intensity_ch2": [5.0] * n,      # constant -> dropped
    })
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert out is df and col == "infected"
    assert "No usable morphology + pathogen features" in capsys.readouterr().out


@pytest.mark.parametrize(
    "n_inf, n_uninf, expect",
    [
        (15, 15, "Not enough cells with finite intensity"),   # < 40 cells total
        (5, 60, "Not enough cells with finite intensity"),    # < 10 infected
        (60, 5, "Not enough cells with finite intensity"),    # < 10 uninfected
    ],
)
def test_qc_ground_truth_too_small(n_inf, n_uninf, expect, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=n_inf, n_uninfected=n_uninf, n_frames=1)
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert out is df and col == "infected"
    assert "adjusted_infected" not in out.columns
    assert expect in capsys.readouterr().out


# ===========================================================================
# _infection_qc_pca_clustering - PCA happy path
# ===========================================================================

def test_qc_pca_relabel_happy_path(tmp_path, capsys, png_figure_preference):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=3)
    # stale adjusted columns must be dropped before the merge
    df["adjusted_infected"] = False
    df["adjusted_infected_x"] = 1.0
    settings = {"infection_intensity_mode": "relabel"}
    motility = tmp_path / "motility"

    out, col = _infection_qc_pca_clustering(
        df, settings, "infected", 2, str(motility)
    )

    assert col == "adjusted_infected"
    assert "adjusted_infected_x" not in out.columns
    assert out["adjusted_infected"].dtype == bool
    assert len(out) == len(df)
    # clean synthetic separation -> clusters recover the ground truth exactly
    assert (out["adjusted_infected"].to_numpy() == out["infected"].to_numpy()).all()

    payload = settings["infection_pca_data"]
    assert payload["method_label"] == "PCA"
    assert payload["coords"].shape == (120, 2)
    assert payload["labels"].shape == (120,)
    assert payload["cluster_labels"].shape == (120,)
    assert payload["embedding_params"] == {}
    assert payload["gt_sep_score"] == pytest.approx(1.0)
    assert payload["silhouette_score"] > 0.5
    assert payload["centroid_distance"] > 0
    assert {payload["infected_cluster"], payload["uninfected_cluster"]} == {0, 1}
    assert payload["initial_infected_frac_infected_cluster"] == pytest.approx(1.0)
    assert payload["initial_infected_frac_uninfected_cluster"] == pytest.approx(0.0)
    assert settings["infection_pca_method"] == "pca"
    assert settings["infection_intensity_qc_panel_type"] == "pca"
    assert settings["infection_intensity_qc_panel_path"] is None

    png = motility / "infection_pca_qc_embedding.png"
    assert png.is_file() and png.stat().st_size > 0
    assert "Relabel mode" in capsys.readouterr().out


def test_qc_pca_unknown_strategy_falls_back_to_pca(tmp_path):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {"infection_intensity_strategy": "magic-embedding"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert col == "adjusted_infected"
    assert settings["infection_pca_method"] == "pca"
    assert settings["infection_pca_data"]["method_label"] == "PCA"


def test_qc_small_ground_truth_warns_but_still_runs(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=22, n_uninfected=22, n_frames=1)
    out, col = _infection_qc_pca_clustering(df, {}, "infected", 2, None)
    assert col == "adjusted_infected"
    txt = capsys.readouterr().out
    assert "Very small ground-truth subsets" in txt
    assert "Ground-truth sets:" in txt


def test_qc_pathogen_weight_and_no_log_transform():
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_pca_log_intensity": False,
        "infection_pca_pathogen_weight": 4.0,
    }
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert col == "adjusted_infected"
    # up-weighting the pathogen features must not destroy the separation
    assert settings["infection_pca_data"]["gt_sep_score"] == pytest.approx(1.0)
    assert (out["adjusted_infected"].to_numpy() == out["infected"].to_numpy()).all()


def test_qc_negative_intensity_skips_log1p():
    """A negative value in an intensity feature disables its log1p transform."""
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1,
                     negative_intensity=True)
    settings = {"infection_pca_log_intensity": True}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert col == "adjusted_infected"
    assert settings["infection_pca_data"]["coords"].shape[0] == 120


def test_qc_nan_features_are_median_imputed():
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    # punch holes in one morphology feature; the remaining features keep every
    # row finite, so the rows survive and the NaNs get median-imputed
    df.loc[df.index[:20], "cell_perimeter"] = np.nan
    settings = {}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert col == "adjusted_infected"
    assert len(out) == 120
    assert out["adjusted_infected"].notna().all()
    # imputation happened in-place: no NaN made it into the embedding
    assert np.isfinite(settings["infection_pca_data"]["coords"]).all()


def test_qc_feature_with_too_few_observations_is_dropped():
    """A feature observed in <10 cells is discarded before the embedding."""
    from spacr.timelapse import _infection_qc_pca_clustering

    base = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)

    sparse = base.copy()
    sparse["cell_perimeter"] = np.nan
    sparse.loc[sparse.index[:5], "cell_perimeter"] = [80.0, 90.0, 100.0, 110.0, 120.0]
    s1 = {}
    _infection_qc_pca_clustering(sparse, s1, "infected", 2, None)

    dropped = base.drop(columns=["cell_perimeter"])
    s2 = {}
    _infection_qc_pca_clustering(dropped, s2, "infected", 2, None)

    # keeping 5 stray observations must be identical to not having the column
    np.testing.assert_allclose(
        s1["infection_pca_data"]["coords"], s2["infection_pca_data"]["coords"]
    )


def test_qc_intensity_column_readded_when_channel_case_mismatches():
    """`pathogen_chan='A'` -> the ch-matching filter misses `..._chA`, so the
    intensity column has to be re-added explicitly; the embedding must match
    the equivalent all-lowercase run exactly."""
    from spacr.timelapse import _infection_qc_pca_clustering

    base = make_all_df(n_infected=60, n_uninfected=60, n_frames=1).drop(
        columns=["cell_mean_intensity_ch2", "cell_mean_intensity_ch0"]
    )

    upper = base.rename(columns={"cell_p95_intensity_ch2": "cell_p95_intensity_chA"})
    s_up = {}
    out_up, col_up = _infection_qc_pca_clustering(upper, s_up, "infected", "A", None)

    lower = base.rename(columns={"cell_p95_intensity_ch2": "cell_p95_intensity_cha"})
    s_lo = {}
    out_lo, col_lo = _infection_qc_pca_clustering(lower, s_lo, "infected", "a", None)

    assert col_up == col_lo == "adjusted_infected"
    np.testing.assert_allclose(
        s_up["infection_pca_data"]["coords"], s_lo["infection_pca_data"]["coords"]
    )
    assert s_up["infection_pca_data"]["gt_sep_score"] == pytest.approx(1.0)


def test_qc_subsampling_leaves_unsampled_cells_with_original_labels(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=2)
    settings = {"infection_pca_max_cells": 60}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert col == "adjusted_infected"
    # only 60 of 120 cells were embedded ...
    assert settings["infection_pca_data"]["coords"].shape == (60, 2)
    # ... but every frame-level row still carries a label
    assert len(out) == len(df)
    assert out["adjusted_infected"].notna().all()
    assert out["adjusted_infected"].dtype == bool


# ===========================================================================
# _infection_qc_pca_clustering - remove mode
# ===========================================================================

def test_qc_remove_mode_drops_disagreeing_cells(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=2, n_mislabeled=8)
    settings = {"infection_intensity_mode": "remove"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert col == "adjusted_infected"
    # the 8 cells whose features contradict their label are gone (2 frames each)
    assert len(out) == len(df) - 8 * 2
    assert out["cellID"].nunique() == 112
    # in remove mode the surviving labels equal the ORIGINAL labels
    assert (out["adjusted_infected"].to_numpy() == out["infected"].to_numpy()).all()
    assert settings["infection_pca_data"]["coords"].shape == (112, 2)
    assert "Remove mode: removed 8 cells" in capsys.readouterr().out


def test_qc_remove_mode_with_no_disagreement_keeps_everything(capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {"infection_intensity_mode": "REMOVE"}  # case-insensitive
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert col == "adjusted_infected"
    assert len(out) == len(df)
    assert "Remove mode: removed" not in capsys.readouterr().out


# ===========================================================================
# _infection_qc_pca_clustering - degenerate clustering (failure injection)
# ===========================================================================

def test_qc_single_cluster_zero_gives_empty_uninfected_cluster(
        tmp_path, capsys, png_figure_preference):
    """KMeans collapsing to one cluster: empty-cluster centroid + 0.5 GT frac."""
    from spacr.timelapse import _infection_qc_pca_clustering
    mp = pytest.MonkeyPatch()
    try:
        _install_const_kmeans(mp, 0)
        df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
        settings = {}
        motility = tmp_path / "mot"
        out, col = _infection_qc_pca_clustering(
            df, settings, "infected", 2, str(motility)
        )
    finally:
        mp.undo()

    payload = settings["infection_pca_data"]
    assert payload["infected_cluster"] == 0
    assert payload["uninfected_cluster"] == 1
    assert payload["gt_sep_score"] == pytest.approx(0.0)
    assert payload["centroid_distance"] > 0          # centroid[1] == origin
    assert payload["silhouette_score"] is None       # only one distinct label
    assert payload["initial_infected_frac_uninfected_cluster"] == 0.0
    assert payload["initial_infected_frac_infected_cluster"] == pytest.approx(0.5)
    # everything landed in the "infected" cluster
    assert bool(out["adjusted_infected"].all())
    txt = capsys.readouterr().out
    assert "WARNING: weak cluster structure" in txt
    assert (motility / "infection_pca_qc_embedding.png").is_file()


def test_qc_single_cluster_one_gives_empty_infected_cluster():
    from spacr.timelapse import _infection_qc_pca_clustering
    mp = pytest.MonkeyPatch()
    try:
        _install_const_kmeans(mp, 1)
        # more uninfected than infected -> GT fraction of cluster 1 < 0.5,
        # so argmax picks the (empty) cluster 0 as "infected"
        df = make_all_df(n_infected=40, n_uninfected=90, n_frames=1)
        settings = {}
        out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    finally:
        mp.undo()

    payload = settings["infection_pca_data"]
    assert payload["infected_cluster"] == 0
    assert payload["uninfected_cluster"] == 1
    assert payload["initial_infected_frac_infected_cluster"] == 0.0
    assert payload["initial_infected_frac_uninfected_cluster"] > 0.0
    assert payload["centroid_distance"] > 0
    assert not bool(out["adjusted_infected"].any())


def test_qc_silhouette_failure_is_swallowed(tmp_path, png_figure_preference):
    """A raising silhouette_score leaves silhouette_score=None in the payload."""
    from spacr.timelapse import _infection_qc_pca_clustering
    import sklearn.metrics

    mp = pytest.MonkeyPatch()

    def boom(*a, **k):
        raise ValueError("synthetic silhouette failure")

    try:
        mp.setattr(sklearn.metrics, "silhouette_score", boom)
        df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
        settings = {}
        motility = tmp_path / "mot_sil"
        out, col = _infection_qc_pca_clustering(
            df, settings, "infected", 2, str(motility)
        )
    finally:
        mp.undo()

    assert settings["infection_pca_data"]["silhouette_score"] is None
    assert col == "adjusted_infected"
    assert (motility / "infection_pca_qc_embedding.png").is_file()


def test_qc_plot_failure_is_reported_not_raised(tmp_path, capsys):
    """motility_dir pointing at a FILE makes os.makedirs blow up."""
    from spacr.timelapse import _infection_qc_pca_clustering

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")

    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, str(blocker))

    assert col == "adjusted_infected"
    assert "Failed to save embedding QC plot" in capsys.readouterr().out
    assert blocker.read_text() == "x"


# ===========================================================================
# _infection_qc_pca_clustering - UMAP branch
# ===========================================================================

def test_qc_umap_unavailable_falls_back_to_pca(monkeypatch, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering
    from spacr import utils

    monkeypatch.setattr(utils, "umap", None)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {"infection_intensity_strategy": "umap"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert settings["infection_pca_method"] == "umap"
    assert settings["infection_pca_data"]["method_label"] == "PCA"
    assert "not available; falling back to PCA" in capsys.readouterr().out


def test_qc_umap_no_search_uses_configured_params(monkeypatch, tmp_path,
                                                  png_figure_preference):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_umap(monkeypatch)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "umap",
        "infection_pca_umap_search": False,
        "infection_pca_umap_n_neighbors": 9,
        "infection_pca_umap_min_dist": 0.25,
    }
    motility = tmp_path / "mot_umap"
    out, col = _infection_qc_pca_clustering(
        df, settings, "infected", 2, str(motility)
    )

    assert calls == [(9, 0.25)]
    payload = settings["infection_pca_data"]
    assert payload["method_label"] == "UMAP"
    assert payload["embedding_params"] == {"n_neighbors": 9, "min_dist": 0.25}
    assert col == "adjusted_infected"
    assert (motility / "infection_umap_qc_embedding.png").is_file()


def test_qc_umap_grid_search_picks_best_and_survives_failures(monkeypatch, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_umap(monkeypatch, fail_when=lambda nn, md: nn == 3)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "umap",
        "infection_pca_umap_n_neighbors_grid": [3, 7],
        "infection_pca_umap_min_dist_grid": [0.0, 0.2],
    }
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert len(calls) == 4                     # every grid point attempted
    params = settings["infection_pca_data"]["embedding_params"]
    # n_neighbors=3 always raised; the larger scale wins on centroid distance
    assert params == {"n_neighbors": 7, "min_dist": 0.0}
    txt = capsys.readouterr().out
    assert "UMAP trial failed for n_neighbors=3" in txt
    assert "UMAP best params" in txt


def test_qc_umap_all_trials_fail_raises(monkeypatch):
    from spacr.timelapse import _infection_qc_pca_clustering

    _install_fake_umap(monkeypatch, fail_when=lambda nn, md: True)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "umap",
        "infection_pca_umap_n_neighbors_grid": [5],
        "infection_pca_umap_min_dist_grid": [0.1],
    }
    with pytest.raises(RuntimeError, match="UMAP hyperparameter search failed"):
        _infection_qc_pca_clustering(df, settings, "infected", 2, None)


# ===========================================================================
# _infection_qc_pca_clustering - t-SNE branch
# ===========================================================================

def test_qc_tsne_unavailable_falls_back_to_pca(monkeypatch, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering
    import sklearn.manifold

    monkeypatch.delattr(sklearn.manifold, "TSNE")
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {"infection_intensity_strategy": "tsne"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    assert settings["infection_pca_method"] == "tsne"
    assert settings["infection_pca_data"]["method_label"] == "PCA"
    assert "not available; falling back to PCA" in capsys.readouterr().out


def test_qc_tsne_no_search_single_run(monkeypatch, tmp_path,
                                      png_figure_preference):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_tsne(monkeypatch)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "tsne",
        "infection_pca_tsne_search": False,
        "infection_pca_tsne_perplexity": 12.0,
    }
    motility = tmp_path / "mot_tsne"
    out, col = _infection_qc_pca_clustering(
        df, settings, "infected", 2, str(motility)
    )

    assert calls == [(12.0, "auto")]
    payload = settings["infection_pca_data"]
    assert payload["method_label"] == "t-SNE"
    assert payload["embedding_params"] == {"perplexity": 12.0, "learning_rate": "auto"}
    assert (motility / "infection_tsne_qc_embedding.png").is_file()


def test_qc_tsne_no_search_nonpositive_perplexity_uses_max(monkeypatch):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_tsne(monkeypatch)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "tsne",
        "infection_pca_tsne_search": False,
        "infection_pca_tsne_perplexity": -3.0,
    }
    _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    max_perp = (120 - 1) / 3.0
    assert calls == [(pytest.approx(max_perp), "auto")]
    assert settings["infection_pca_data"]["embedding_params"]["perplexity"] == \
        pytest.approx(max_perp)


def test_qc_tsne_grid_search_picks_highest_score(monkeypatch, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_tsne(monkeypatch)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {"infection_intensity_strategy": "tsne"}
    out, col = _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    # default perplexity grid is [15, 30, 45]; max_perp = 119/3 = 39.67
    # so 45 is filtered out and 2 learning rates are tried for each survivor
    assert [p for p, _ in calls] == [15.0, 15.0, 30.0, 30.0]
    params = settings["infection_pca_data"]["embedding_params"]
    assert params == {"perplexity": 30.0, "learning_rate": 200.0}
    assert "t-SNE best params" in capsys.readouterr().out


def test_qc_tsne_all_perplexities_filtered_out(monkeypatch):
    from spacr.timelapse import _infection_qc_pca_clustering

    calls = _install_fake_tsne(monkeypatch)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "tsne",
        "infection_pca_tsne_perplexity_grid": [500.0, 0.0],
        "infection_pca_tsne_learning_rate_grid": [100.0],
    }
    _infection_qc_pca_clustering(df, settings, "infected", 2, None)

    # nothing in the grid is usable -> single fallback candidate min(30, max_perp)
    assert calls == [(30.0, 100.0)]
    assert settings["infection_pca_data"]["embedding_params"] == {
        "perplexity": 30.0, "learning_rate": 100.0,
    }


def test_qc_tsne_all_trials_fail_raises(monkeypatch, capsys):
    from spacr.timelapse import _infection_qc_pca_clustering

    _install_fake_tsne(monkeypatch, always_fail=True)
    df = make_all_df(n_infected=60, n_uninfected=60, n_frames=1)
    settings = {
        "infection_intensity_strategy": "tsne",
        "infection_pca_tsne_perplexity_grid": [10.0],
        "infection_pca_tsne_learning_rate_grid": [200.0],
    }
    with pytest.raises(RuntimeError, match="t-SNE hyperparameter search failed"):
        _infection_qc_pca_clustering(df, settings, "infected", 2, None)
    assert "t-SNE trial failed" in capsys.readouterr().out
