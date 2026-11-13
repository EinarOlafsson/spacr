"""CPU coverage for the spacr.utils UMAP / embedding-plotting block.

Covers the DB-path helpers, ``load_image_paths``, ``filter_columns``,
``reduction_and_clustering`` (float / clamped neighbour counts, transform
mode, missing-model guard) and the matplotlib cluster/grid plotting helpers
including their degenerate-geometry fallbacks.
"""
from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# get_db_paths / get_sequencing_paths
# ---------------------------------------------------------------------------

def test_get_db_paths_and_sequencing_paths_wrap_scalar_and_list():
    from spacr.utils import get_db_paths, get_sequencing_paths

    join = os.path.join
    single = get_db_paths("/root/exp")
    assert single == [join("/root/exp", "measurements", "measurements.db")]

    multi = get_db_paths(["/a", "/b"])
    assert multi == [
        join("/a", "measurements", "measurements.db"),
        join("/b", "measurements", "measurements.db"),
    ]

    seq = get_sequencing_paths("/root/exp")
    assert seq == [join("/root/exp", "sequencing", "sequencing_data.csv")]
    assert get_sequencing_paths(["/a", "/b"]) == [
        join("/a", "sequencing", "sequencing_data.csv"),
        join("/b", "sequencing", "sequencing_data.csv"),
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


# ---------------------------------------------------------------------------
# reduction_and_clustering
# ---------------------------------------------------------------------------

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

    from matplotlib.colors import to_rgba

    fig, ax = setup_plot(4, black_background)
    expected = "black" if black_background else "white"
    ink = "white" if black_background else "black"

    # ON THE FIGURE, NOT ON `plt.rcParams`. This asserted the process-wide
    # values after the call -- which was asserting a LEAK: rcParams is shared,
    # so theming one figure through it themed every LATER figure in the
    # process, including one being saved for paper. Instruction 136 asks by
    # name that no module write rcParams globally, and `setup_plot` now scopes
    # them with `rc_context`.
    #
    # Note this passed for black_background=False only because white is
    # already matplotlib's default -- the assertion was never measuring the
    # theming.
    assert fig.get_facecolor() == to_rgba(expected)
    assert ax.get_facecolor() == to_rgba(expected)
    assert to_rgba(ax.xaxis.label.get_color()) == to_rgba(ink)
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


# ---------------------------------------------------------------------------
# plot_embedding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# plot_clusters_grid / plot_grid
# ---------------------------------------------------------------------------
