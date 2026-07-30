"""CPU coverage for the spacr.utils UMAP / embedding-plotting block.

Covers the DB-path helpers, ``load_image_paths``, ``filter_columns``,
``reduction_and_clustering`` (float / clamped neighbour counts, transform
mode, missing-model guard) and the matplotlib cluster/grid plotting helpers
including their degenerate-geometry fallbacks.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate, and keep setup_plot's global
    rcParams mutation from leaking into other test modules."""
    saved = matplotlib.rcParams.copy()
    yield
    plt.close("all")
    matplotlib.rcParams.update(saved)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png_list_db(tmp_path):
    """A sqlite DB whose ``png_list`` table has cell and nucleus PNG rows."""
    db_path = tmp_path / "measurements.db"
    df = pd.DataFrame(
        {
            "prcfo": ["p1_A01_1_o1", "p1_A01_1_o2", "p1_A02_1_o1", "p1_A02_1_o2"],
            "png_path": [
                "/data/p1/cell_png/p1_A01_1_o1.png",
                "/data/p1/nucleus_png/p1_A01_1_o2.png",
                "/data/p1/cell_png/p1_A02_1_o1.png",
                "/data/p1/nucleus_png/p1_A02_1_o2.png",
            ],
            "object_label": [1, 2, 1, 2],
        }
    )
    con = sqlite3.connect(db_path)
    try:
        df.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db_path


def _write_gray_pngs(tmp_path, n, size=8):
    """Write ``n`` non-empty 8-bit grayscale PNGs and return their paths."""
    paths = []
    for i in range(n):
        arr = np.zeros((size, size), np.uint8)
        arr[2:6, 2:6] = 50 + 10 * i          # non-zero -> remove_canvas is safe
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr, mode="L").save(p)
        paths.append(str(p))
    return paths


def _blob_points(n=12, seed=0):
    """A 2-D point cloud in general position (a proper 2-D convex hull)."""
    rs = np.random.RandomState(seed)
    return rs.uniform(-1.0, 1.0, size=(n, 2)).astype(float)


def _collinear_points(n=5):
    """``n`` exactly collinear points -> Qhull cannot build a 2-D hull."""
    return np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)])


# ---------------------------------------------------------------------------
# get_db_paths / get_sequencing_paths
# ---------------------------------------------------------------------------

def test_get_db_paths_and_sequencing_paths_wrap_scalar_and_list():
    from spacr.utils import get_db_paths, get_sequencing_paths

    single = get_db_paths("/root/exp")
    assert single == ["/root/exp/measurements/measurements.db"]

    multi = get_db_paths(["/a", "/b"])
    assert multi == [
        "/a/measurements/measurements.db",
        "/b/measurements/measurements.db",
    ]

    seq = get_sequencing_paths("/root/exp")
    assert seq == ["/root/exp/sequencing/sequencing_data.csv"]
    assert get_sequencing_paths(["/a", "/b"]) == [
        "/a/sequencing/sequencing_data.csv",
        "/b/sequencing/sequencing_data.csv",
    ]


# ---------------------------------------------------------------------------
# load_image_paths
# ---------------------------------------------------------------------------

def test_load_image_paths_returns_all_rows_indexed_by_prcfo(tmp_path):
    from spacr.utils import load_image_paths

    db_path = _png_list_db(tmp_path)
    con = sqlite3.connect(db_path)
    try:
        out = load_image_paths(con.cursor(), visualize=None)
    finally:
        con.close()

    assert isinstance(out, pd.DataFrame)
    assert out.index.name == "prcfo"
    assert len(out) == 4
    assert list(out.columns) == ["png_path", "object_label"]
    assert out.loc["p1_A01_1_o1", "png_path"].endswith("cell_png/p1_A01_1_o1.png")


def test_load_image_paths_filters_on_object_type(tmp_path):
    from spacr.utils import load_image_paths

    db_path = _png_list_db(tmp_path)
    con = sqlite3.connect(db_path)
    try:
        cells = load_image_paths(con.cursor(), visualize="cell")
        nuclei = load_image_paths(con.cursor(), visualize="nucleus")
    finally:
        con.close()

    assert sorted(cells.index) == ["p1_A01_1_o1", "p1_A02_1_o1"]
    assert cells["png_path"].str.contains("cell_png").all()
    assert sorted(nuclei.index) == ["p1_A01_1_o2", "p1_A02_1_o2"]
    assert not nuclei["png_path"].str.contains("cell_png").any()


# ---------------------------------------------------------------------------
# merge_dataframes / filter_columns
# ---------------------------------------------------------------------------

def test_merge_dataframes_joins_on_prcfo_index():
    from spacr.utils import merge_dataframes

    image_paths_df = pd.DataFrame(
        {"png_path": ["a.png", "b.png"]}, index=pd.Index(["x", "y"], name="prcfo")
    )
    df = pd.DataFrame({"prcfo": ["y", "z"], "cell_area": [10.0, 20.0]})

    out = merge_dataframes(df, image_paths_df, verbose=False)

    # inner join -> only the shared 'y' key survives
    assert list(out.index) == ["y"]
    assert out.loc["y", "png_path"] == "b.png"
    assert out.loc["y", "cell_area"] == 10.0
    assert list(out.columns) == ["png_path", "cell_area"]


def test_merge_dataframes_verbose_displays_result():
    from spacr.utils import merge_dataframes

    image_paths_df = pd.DataFrame(
        {"png_path": ["a.png", "b.png"]}, index=pd.Index(["x", "y"], name="prcfo")
    )
    df = pd.DataFrame({"prcfo": ["x", "y"], "cell_area": [1.0, 2.0]})

    out = merge_dataframes(df, image_paths_df, verbose=True)

    assert len(out) == 2
    assert out["cell_area"].tolist() == [1.0, 2.0]


def test_filter_columns_by_substring():
    from spacr.utils import filter_columns

    df = pd.DataFrame(
        {
            "cell_channel_0_mean": [1.0],
            "cell_channel_1_mean": [2.0],
            "cell_area": [3.0],
        }
    )
    out = filter_columns(df, filter_by="channel_1")
    assert list(out.columns) == ["cell_channel_1_mean"]
    assert out.iloc[0, 0] == 2.0


def test_filter_columns_morphology_drops_channel_columns():
    """The 'morphology' sentinel keeps every column WITHOUT 'channel'."""
    from spacr.utils import filter_columns

    df = pd.DataFrame(
        {
            "cell_area": [3.0],
            "cell_channel_0_mean": [1.0],
            "nucleus_perimeter": [4.0],
            "pathogen_channel_2_max": [5.0],
        }
    )
    out = filter_columns(df, filter_by="morphology")
    assert list(out.columns) == ["cell_area", "nucleus_perimeter"]
    assert out.shape == (1, 2)


def test_filter_columns_morphology_handles_non_string_columns():
    from spacr.utils import filter_columns

    df = pd.DataFrame({0: [1.0], "cell_channel_0": [2.0], 7: [3.0]})
    out = filter_columns(df, filter_by="morphology")
    assert list(out.columns) == [0, 7]


# ---------------------------------------------------------------------------
# reduction_and_clustering
# ---------------------------------------------------------------------------

def _numeric_blob(n=30, d=4, seed=1):
    rs = np.random.RandomState(seed)
    a = rs.normal(0.0, 1.0, size=(n // 2, d))
    b = rs.normal(6.0, 1.0, size=(n - n // 2, d))
    return np.vstack([a, b])


def test_reduction_and_clustering_float_n_neighbors_is_a_fraction_of_n():
    """A float n_neighbors means 'this fraction of the samples'."""
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=30)
    embedding, labels, reducer = reduction_and_clustering(
        data,
        n_neighbors=0.3,          # -> int(0.3 * 30) == 9
        min_dist=0.1,
        metric="euclidean",
        eps=0.5,
        min_samples=2,
        clustering="kmeans",
        reduction_method="tsne",
        verbose=False,
    )
    assert reducer.perplexity == 9
    assert embedding.shape == (30, 2)
    assert labels.shape == (30,)
    assert set(np.unique(labels)) == {0, 1}


def test_reduction_and_clustering_clamps_small_n_neighbors_to_two():
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=20)
    embedding, labels, reducer = reduction_and_clustering(
        data,
        n_neighbors=1,            # <= 2 -> clamped to 2
        min_dist=0.1,
        metric="euclidean",
        eps=0.5,
        min_samples=2,
        clustering="kmeans",
        reduction_method="tsne",
        verbose=False,
    )
    assert reducer.perplexity == 2
    assert embedding.shape == (20, 2)


def test_reduction_and_clustering_float_n_neighbors_also_clamps_to_two():
    """A tiny fraction rounds below 2 and is clamped by the same guard."""
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=10)
    _, _, reducer = reduction_and_clustering(
        data,
        n_neighbors=0.05,         # int(0.05 * 10) == 0 -> clamped to 2
        min_dist=0.1,
        metric="euclidean",
        eps=0.5,
        min_samples=2,
        clustering="kmeans",
        reduction_method="tsne",
        verbose=False,
    )
    assert reducer.perplexity == 2


def test_reduction_and_clustering_verbose_fit_reports_progress(capsys):
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=20)
    embedding, labels, _ = reduction_and_clustering(
        data,
        n_neighbors=5,
        min_dist=0.1,
        metric="euclidean",
        eps=5.0,
        min_samples=2,
        clustering="dbscan",
        reduction_method="tsne",
        verbose=True,
        n_jobs=1,
    )
    out = capsys.readouterr().out
    assert "Trained and fit reducer" in out
    assert "Embedding shape: (20, 2)" in out
    assert embedding.shape == (20, 2)
    assert labels.shape == (20,)


def test_reduction_and_clustering_umap_fit_uses_requested_neighbors():
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=30, seed=11)
    embedding, labels, reducer = reduction_and_clustering(
        data,
        n_neighbors=5,
        min_dist=0.1,
        metric="euclidean",
        eps=0.5,
        min_samples=2,
        clustering="kmeans",
        reduction_method="umap",
        verbose=False,
        n_jobs=1,
    )
    assert reducer.n_neighbors == 5
    assert reducer.n_components == 2
    assert embedding.shape == (30, 2)
    assert set(np.unique(labels)) == {0, 1}


def test_reduction_and_clustering_rejects_unknown_reduction_method():
    from spacr.utils import reduction_and_clustering

    with pytest.raises(ValueError, match="Unsupported reduction method: pca"):
        reduction_and_clustering(
            np.zeros((6, 3)),
            n_neighbors=3,
            min_dist=0.1,
            metric="euclidean",
            eps=0.5,
            min_samples=2,
            clustering="dbscan",
            reduction_method="pca",
        )


class _StubReducer:
    """Minimal 'already fitted' reducer usable in transform mode."""

    def __init__(self):
        self.seen = None

    def transform(self, x):
        self.seen = np.asarray(x)
        return np.asarray(x)[:, :2].astype(float)


def test_reduction_and_clustering_transform_mode_reuses_model(capsys):
    from spacr.utils import reduction_and_clustering

    data = _numeric_blob(n=20, d=4, seed=3)
    model = _StubReducer()

    embedding, labels, reducer = reduction_and_clustering(
        data,
        n_neighbors=5,
        min_dist=0.1,
        metric="euclidean",
        eps=1.0,
        min_samples=3,
        clustering="dbscan",
        reduction_method="umap",
        verbose=True,
        mode="transform",
        model=model,
        n_jobs=1,
    )

    out = capsys.readouterr().out
    assert "Fit data to reducer" in out
    assert "Embedding shape: (20, 2)" in out
    # the model was used verbatim, no new reducer was fitted
    assert reducer is model
    assert model.seen.shape == (20, 4)
    np.testing.assert_allclose(embedding, data[:, :2])
    assert labels.shape == (20,)
    # dbscan on two well-separated gaussians must find at least one cluster
    assert set(np.unique(labels)) - {-1}


def test_reduction_and_clustering_transform_mode_without_model_raises():
    from spacr.utils import reduction_and_clustering

    with pytest.raises(ValueError, match="Model is None"):
        reduction_and_clustering(
            np.zeros((5, 3)),
            n_neighbors=3,
            min_dist=0.1,
            metric="euclidean",
            eps=0.5,
            min_samples=2,
            clustering="dbscan",
            mode="transform",
            model=None,
        )


def test_reduction_and_clustering_transform_mode_default_model_raises_valueerror():
    from spacr.utils import reduction_and_clustering

    # The docstring promises ":raises ValueError: ... missing model".
    with pytest.raises(ValueError):
        reduction_and_clustering(
            np.zeros((5, 3)),
            n_neighbors=3,
            min_dist=0.1,
            metric="euclidean",
            eps=0.5,
            min_samples=2,
            clustering="dbscan",
            mode="transform",   # model left at its default of False
        )


def test_reduction_and_clustering_rejects_unknown_clustering():
    from spacr.utils import reduction_and_clustering

    with pytest.raises(ValueError):
        reduction_and_clustering(
            _numeric_blob(n=8),
            n_neighbors=3,
            min_dist=0.1,
            metric="euclidean",
            eps=0.5,
            min_samples=2,
            clustering="hdbscan",
            reduction_method="tsne",
        )


def test_remove_noise_drops_noise_rows():
    from spacr.utils import remove_noise

    embedding = np.arange(10, dtype=float).reshape(5, 2)
    labels = np.array([0, -1, 1, -1, 1])
    emb2, lab2 = remove_noise(embedding, labels)
    assert emb2.shape == (3, 2)
    np.testing.assert_array_equal(lab2, np.array([0, 1, 1]))
    np.testing.assert_array_equal(emb2, embedding[[0, 2, 4]])


# ---------------------------------------------------------------------------
# colour helpers / setup_plot
# ---------------------------------------------------------------------------

def test_generate_colors_is_deterministic_viridis_for_every_background():
    from spacr.utils import generate_colors

    dark = generate_colors(6, black_background=True)
    light = generate_colors(6, black_background=False)
    assert dark.shape == (6, 4)
    assert light.shape == (6, 4)
    np.testing.assert_allclose(dark, light)
    np.testing.assert_allclose(
        dark[0], matplotlib.colormaps["viridis"](0.08))
    np.testing.assert_allclose(
        dark[-1], matplotlib.colormaps["viridis"](0.92))
    assert np.all(dark[:, 3] == 1)


def test_assign_colors_maps_every_unique_label():
    from spacr.utils import assign_colors

    labels = np.array([-1, 0, 3])
    palette = np.array([[10, 20, 30, 255], [40, 50, 60, 255], [70, 80, 90, 255]], float)
    colors, mapping = assign_colors(labels, palette)
    assert mapping == {-1: 0, 0: 1, 3: 2}
    assert colors[0] == (10.0, 20.0, 30.0, 255.0)
    assert len(colors) == 3


@pytest.mark.parametrize("black_background", [True, False])
def test_setup_plot_applies_theme(black_background):
    from spacr.utils import setup_plot

    fig, ax = setup_plot(4, black_background)
    expected = "black" if black_background else "white"
    assert plt.rcParams["figure.facecolor"] == expected
    assert plt.rcParams["axes.facecolor"] == expected
    assert plt.rcParams["text.color"] == ("white" if black_background else "black")
    assert fig.get_size_inches().tolist() == [4.0, 4.0]
    assert ax.figure is fig


def test_setup_plot_uses_gui_container_colors_and_visible_axes():
    from matplotlib.colors import to_rgba
    from spacr.utils import setup_plot

    colors = {
        "background": "#161719",
        "foreground": "#ffffff",
        "border": "#ffffff",
    }
    fig, ax = setup_plot(
        4, black_background=False, theme_colors=colors)

    assert fig.get_facecolor() == to_rgba("#161719")
    assert ax.get_facecolor() == to_rgba("#161719")
    assert ax.spines["left"].get_edgecolor() == to_rgba("#ffffff")
    assert ax.xaxis.label.get_color() == "#ffffff"
    assert ax.yaxis.label.get_color() == "#ffffff"


# ---------------------------------------------------------------------------
# plot_clusters branches
# ---------------------------------------------------------------------------

def _one_cluster_args(points):
    labels = np.zeros(len(points), dtype=int)
    colors = [(0.5, 0.2, 0.3, 1.0)]
    centers = [points.mean(axis=0)]
    return labels, colors, centers


def test_plot_clusters_smooth_outline_draws_spline():
    from spacr.utils import plot_clusters

    pts = _blob_points(14, seed=5)
    labels, colors, centers = _one_cluster_args(pts)
    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=True, plot_points=True, smooth_lines=True,
        figuresize=10, dot_size=20,
    )
    assert len(ax.lines) == 1
    # smooth_hull_lines evaluates the spline at exactly 100 samples
    assert ax.lines[0].get_xdata().shape == (100,)
    assert ax.lines[0].get_linewidth() == 1
    assert len(ax.collections) == 1


def test_plot_clusters_smooth_outline_swallows_degenerate_hull():
    """Collinear cluster -> Qhull raises inside the try; no outline, no crash."""
    from spacr.utils import plot_clusters

    pts = _collinear_points(6)
    labels, colors, centers = _one_cluster_args(pts)
    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=True, plot_points=True, smooth_lines=True,
    )
    assert len(ax.lines) == 0          # the exception branch was taken
    assert len(ax.collections) == 1    # points still plotted


def test_plot_clusters_convex_hull_outline_draws_one_line_per_simplex():
    from spacr.utils import plot_clusters
    from scipy.spatial import ConvexHull

    pts = _blob_points(14, seed=7)
    labels, colors, centers = _one_cluster_args(pts)
    n_simplices = len(ConvexHull(pts).simplices)
    assert n_simplices >= 3

    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=True, plot_points=True, smooth_lines=False,
    )
    assert len(ax.lines) == n_simplices
    assert all(line.get_linewidth() == 1 for line in ax.lines)


def test_plot_clusters_convex_hull_swallows_degenerate_hull():
    from spacr.utils import plot_clusters

    pts = _collinear_points(7)
    labels, colors, centers = _one_cluster_args(pts)
    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=True, plot_points=True, smooth_lines=False,
    )
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_clusters_without_points_uses_invisible_scatter():
    from spacr.utils import plot_clusters

    pts = _blob_points(10, seed=9)
    labels, colors, centers = _one_cluster_args(pts)
    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=False, plot_points=False, smooth_lines=False,
        figuresize=8, dot_size=33,
    )
    assert len(ax.collections) == 1
    assert ax.collections[0].get_alpha() == 0
    assert ax.collections[0].get_sizes().tolist() == [33]
    # the centroid annotation is written regardless
    assert [t.get_text() for t in ax.texts] == ["0"]
    assert ax.get_xlabel() == "UMAP Dimension 1"
    assert ax.get_ylabel() == "UMAP Dimension 2"


def test_plot_clusters_skips_outline_for_tiny_cluster():
    """A 2-point cluster has no hull; the size guard short-circuits."""
    from spacr.utils import plot_clusters

    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels, colors, centers = _one_cluster_args(pts)
    fig, ax = plt.subplots()
    plot_clusters(
        ax, pts, labels, colors, centers,
        plot_outlines=True, plot_points=True, smooth_lines=True,
    )
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


# ---------------------------------------------------------------------------
# image overlay helpers
# ---------------------------------------------------------------------------

def _count_annotation_boxes(ax):
    return sum(isinstance(c, AnnotationBbox) for c in ax.get_children())


def test_plot_images_by_cluster_skips_the_noise_label(tmp_path):
    """cluster_label == -1 is skipped even when present in cluster_indices."""
    from spacr.utils import plot_images_by_cluster

    paths = _write_gray_pngs(tmp_path, 5)
    embedding = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.2], [5.0, 5.0], [5.2, 5.1]])
    labels = np.array([-1, 0, 0, 1, 1])
    cluster_indices = {-1: np.array([0]), 0: np.array([1, 2]), 1: np.array([3, 4])}
    colors = [(0.1, 0.1, 0.1, 1.0), (0.2, 0.2, 0.2, 1.0), (0.3, 0.3, 0.3, 1.0)]

    fig, ax = plt.subplots()
    plot_images_by_cluster(
        ax, paths, embedding, labels, image_nr=1, img_zoom=0.5,
        colors=colors, cluster_indices=cluster_indices,
        remove_image_canvas=True, verbose=False,
    )
    # one image for cluster 0 and one for cluster 1; nothing for the -1 entry
    assert _count_annotation_boxes(ax) == 2


def test_plot_umap_images_by_cluster_and_random(tmp_path):
    from spacr.utils import plot_umap_images

    paths = _write_gray_pngs(tmp_path, 6)
    embedding = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([-1, 0, 0, 0, 1, 1])
    colors = [(0.1, 0.1, 0.1, 1.0), (0.2, 0.2, 0.2, 1.0), (0.3, 0.3, 0.3, 1.0)]

    fig, ax = plt.subplots()
    plot_umap_images(
        ax, paths, embedding, labels, image_nr=2, img_zoom=0.4,
        colors=colors, plot_by_cluster=True, remove_image_canvas=False,
        verbose=False,
    )
    # cluster 0 has 3 members capped at 2, cluster 1 has exactly 2 -> 4 images
    assert _count_annotation_boxes(ax) == 4

    fig2, ax2 = plt.subplots()
    plot_umap_images(
        ax2, paths, embedding, labels, image_nr=3, img_zoom=0.4,
        colors=colors, plot_by_cluster=False, remove_image_canvas=False,
        verbose=False,
    )
    assert _count_annotation_boxes(ax2) == 3


def test_plot_image_with_and_without_canvas_removal(tmp_path):
    from spacr.utils import plot_image

    path = _write_gray_pngs(tmp_path, 1)[0]
    img = Image.open(path)

    fig, ax = plt.subplots()
    plot_image(ax, 0.5, 0.5, img, img_zoom=0.3, remove_image_canvas=True)
    boxes = [c for c in ax.get_children() if isinstance(c, AnnotationBbox)]
    assert len(boxes) == 1
    # remove_canvas returns an RGBA float array
    assert boxes[0].offsetbox.get_data().shape[-1] == 4

    fig2, ax2 = plt.subplots()
    plot_image(ax2, 0.5, 0.5, Image.open(path), img_zoom=0.3,
               remove_image_canvas=False)
    boxes2 = [c for c in ax2.get_children() if isinstance(c, AnnotationBbox)]
    assert boxes2[0].offsetbox.get_data().ndim == 2


def test_remove_canvas_modes_and_error():
    from spacr.utils import remove_canvas

    gray = np.zeros((4, 4), np.uint8)
    gray[1:3, 1:3] = 200
    out = remove_canvas(Image.fromarray(gray, mode="L"))
    assert out.shape == (4, 4, 4)
    assert out[..., 3].sum() == 4          # only the non-zero pixels are opaque
    assert out[1, 1, 0] == pytest.approx(1.0)

    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    out_rgb = remove_canvas(Image.fromarray(rgb, mode="RGB"))
    assert out_rgb.shape == (4, 4, 4)
    assert out_rgb[0, 0, 3] == 1.0
    assert out_rgb[3, 3, 3] == 0.0

    with pytest.raises(ValueError, match="Unsupported image mode"):
        remove_canvas(Image.fromarray(np.zeros((4, 4, 4), np.uint8), mode="RGBA"))


# ---------------------------------------------------------------------------
# plot_embedding
# ---------------------------------------------------------------------------

def test_plot_embedding_returns_figure_with_images(tmp_path):
    from spacr.utils import plot_embedding

    paths = _write_gray_pngs(tmp_path, 6)
    embedding = np.array(
        [[0.0, 0.0], [0.4, 0.1], [0.1, 0.5], [4.0, 4.0], [4.3, 4.1], [4.1, 4.4]]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    colors = np.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]])

    fig = plot_embedding(
        embedding, paths, labels, image_nr=1, img_zoom=0.3, colors=colors,
        plot_by_cluster=True, plot_outlines=True, plot_points=True,
        plot_images=True, smooth_lines=False, black_background=False,
        figuresize=5, dot_size=20, remove_image_canvas=False, verbose=False,
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.collections) == 2                 # one scatter per cluster
    assert _count_annotation_boxes(ax) == 2         # one image per cluster
    assert sorted(t.get_text() for t in ax.texts) == ["0", "1"]


def test_plot_embedding_without_images_skips_overlay():
    from spacr.utils import plot_embedding

    embedding = np.array([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [5.0, 5.0]])
    labels = np.array([0, 0, 0, 1])
    colors = np.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]])

    fig = plot_embedding(
        embedding, None, labels, image_nr=1, img_zoom=0.3, colors=colors,
        plot_by_cluster=False, plot_outlines=False, plot_points=True,
        plot_images=True, smooth_lines=False, black_background=True,
        figuresize=4, dot_size=10, remove_image_canvas=True, verbose=False,
    )
    ax = fig.axes[0]
    assert _count_annotation_boxes(ax) == 0
    assert len(ax.collections) == 2


# ---------------------------------------------------------------------------
# plot_clusters_grid / plot_grid
# ---------------------------------------------------------------------------

def test_plot_clusters_grid_returns_none_when_only_noise(capsys):
    from spacr.utils import plot_clusters_grid

    labels = np.array([-1, -1, -1])
    out = plot_clusters_grid(
        embedding=np.zeros((3, 2)), labels=labels, image_nr=2,
        image_paths=["a.png", "b.png", "c.png"],
        colors=np.array([[0.1, 0.2, 0.3, 1.0]]), figuresize=4,
        black_background=False, verbose=False,
    )
    assert out is None
    assert "No clusters found." in capsys.readouterr().out
    # nothing was drawn
    assert plt.get_fignums() == []


def test_plot_clusters_grid_builds_one_axes_per_cluster(tmp_path):
    from spacr.utils import plot_clusters_grid

    paths = _write_gray_pngs(tmp_path, 7)
    labels = np.array([-1, 0, 0, 0, 1, 1, 1])
    colors = np.array(
        [[0.9, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]]
    )
    fig = plot_clusters_grid(
        embedding=np.zeros((7, 2)), labels=labels, image_nr=2,
        image_paths=paths, colors=colors, figuresize=3,
        black_background=False, verbose=False,
    )
    assert isinstance(fig, plt.Figure)
    # two clusters (noise excluded) -> two panels, each with 2 inset images
    top_axes = fig.axes
    assert len(top_axes) == 2
    assert [len(a.child_axes) for a in top_axes] == [2, 2]
    assert len(fig.texts) == 2
    assert {t.get_text() for t in fig.texts} == {"Cluster 0", "Cluster 1"}


def test_plot_grid_single_cluster_and_figsize_cap():
    from spacr.utils import plot_grid

    img = np.full((6, 6), 100, np.uint8)
    fig = plot_grid(
        {0: [img, img, img]},
        colors=np.array([[0.2, 0.4, 0.6, 1.0]]),
        figuresize=500,           # 500 * 1 > 200 -> clamped to 200
        black_background=True,
        verbose=False,
    )
    assert fig.get_size_inches().tolist() == [200.0, 200.0]
    assert len(fig.axes) == 1                 # a single panel...
    assert len(fig.axes[0].child_axes) == 3   # ...with one inset per image
    assert fig.axes[0].axison is False         # axes.axis('off')
    assert all(a.axison is False for a in fig.axes[0].child_axes)


def test_plot_grid_string_cluster_labels_index_colors_positionally(capsys):
    from spacr.utils import plot_grid

    img = np.full((5, 5), 30, np.uint8)
    colors = np.array([[0.1, 0.1, 0.1, 1.0], [0.7, 0.7, 0.7, 1.0]])
    fig = plot_grid(
        {"alpha": [img], "beta": [img, img]},
        colors=colors, figuresize=2, black_background=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "Lable: alpha index: 0" in out
    assert "Lable: beta index: 1" in out
    assert {t.get_text() for t in fig.texts} == {"Cluster alpha", "Cluster beta"}
    assert fig.get_size_inches().tolist() == [4.0, 2.0]
