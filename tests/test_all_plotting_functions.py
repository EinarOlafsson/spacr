"""Coverage for spacr.plot — exercise the public plotting functions with
synthetic inputs on the Agg backend.

Each test builds the minimal real-shaped input a function needs and asserts
on what came out: how many panels, what they were drawn from, what landed on
disk. "It ran without raising" is not an assertion — fourteen tests in here
used to call the function and check nothing at all, which means they could
not tell a correct plot from a blank canvas, a plot of the wrong column, or
a save that wrote a 0-byte file. Matplotlib is happy to draw all three.

The rules this file holds itself to:

* No test without an assertion, and no ``assert fig is not None`` standing in
  for one. Assert the panel count, the data the artists were built from, the
  titles the user reads, and the size of anything saved.
* Nothing may swallow an exception into a ``pytest.skip``: a self-skip makes a
  broken function and an unsupported fixture look identical, and every skip
  this file used to carry was hiding either a product bug or a wrong call.
  Build the real input, or assert the real failure with xfail(strict=True).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _synth_mask(size=64, n=6):
    m = np.zeros((size, size), dtype=np.int32)
    rng = np.random.default_rng(0)
    for lbl in range(1, n + 1):
        cy, cx = rng.integers(8, size - 8, size=2)
        y, x = np.ogrid[:size, :size]
        m[(x - cx) ** 2 + (y - cy) ** 2 < 25] = lbl
    return m


def _imshow_arrays(ax):
    """The arrays actually handed to imshow on ``ax``, in draw order.

    Asserting on these is the difference between "a figure exists" and "the
    figure shows the data that was passed in".
    """
    return [np.asarray(im.get_array()) for im in ax.get_images()]


def _nonempty_file(path, minimum=1000):
    """A saved figure exists and is not a blank/0-byte stub."""
    path = Path(path)
    assert path.is_file(), f"{path} was not written"
    size = path.stat().st_size
    assert size > minimum, f"{path.name} is only {size} bytes -- looks blank"
    return size


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def test_generate_mask_random_cmap():
    """One colour per label, and label 0 is opaque black.

    Background must be black: it is drawn under every object, so a random
    colour there makes an empty field look segmented.
    """
    mask = _synth_mask(n=6)
    cmap = P.generate_mask_random_cmap(mask)
    assert cmap.N == 7, "6 objects + background"
    assert tuple(cmap(0)) == (0.0, 0.0, 0.0, 1.0)
    colours = np.array([cmap(i) for i in range(1, cmap.N)])
    assert (colours[:, 3] == 1).all(), "object colours must be opaque"
    assert len(np.unique(colours, axis=0)) == len(colours), (
        "two objects were given the same colour")


def test_random_cmap():
    cmap = P.random_cmap(50)
    assert cmap.N == 51
    assert tuple(cmap(0)) == (0.0, 0.0, 0.0, 1.0)
    assert all(cmap(i)[3] == 1 for i in range(cmap.N))


# ---------------------------------------------------------------------------
# Mask / flow visualisers
# ---------------------------------------------------------------------------

def test_visualize_masks():
    """Three panels, each showing its own mask, under the caller's suptitle.

    The suptitle is asserted because the loop variable used to be named
    ``title`` and shadowed the parameter, so every figure read 'Mask 3'.
    """
    # Three DIFFERENT masks (6/4/2 objects): identical ones would let a
    # panel-swap pass unnoticed, which is the whole thing this figure is for.
    m1, m2, m3 = _synth_mask(n=6), _synth_mask(n=4), _synth_mask(n=2)
    P.visualize_masks(m1, m2, m3, title="my comparison")

    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == ["Mask 1", "Mask 2", "Mask 3"]
    assert fig._suptitle.get_text() == "my comparison"
    for ax, mask in zip(fig.axes, (m1, m2, m3)):
        drawn = _imshow_arrays(ax)
        assert len(drawn) == 1
        assert np.array_equal(drawn[0], mask)
    assert [_imshow_arrays(ax)[0].max() for ax in fig.axes] == [6, 4, 2]


def test_visualize_cellpose_masks():
    masks = [_synth_mask(), _synth_mask(n=2)]
    P.visualize_cellpose_masks(masks, titles=["a", "b"], filename="field_007")

    fig = plt.gcf()
    assert len(fig.axes) == len(masks), "one panel per mask"
    assert [ax.get_title() for ax in fig.axes] == ["a", "b"]
    # The suptitle names the field, which is the only thing tying the figure
    # to the data on screen.
    assert "field_007" in fig._suptitle.get_text()
    for ax, mask in zip(fig.axes, masks):
        assert np.array_equal(_imshow_arrays(ax)[0], mask)


def test_visualize_cellpose_masks_saves_a_pdf(tmp_path):
    masks = [_synth_mask()]
    P.visualize_cellpose_masks(masks, filename="field_001", save=True,
                                 src=str(tmp_path))
    written = list(Path(tmp_path).rglob("*.pdf"))
    assert len(written) == 1, f"expected one PDF, got {written}"
    _nonempty_file(written[0])


def test_normalize_and_visualize():
    """Two panels: the raw image and the normalized one, side by side."""
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    normalized = (img - img.min()) / (img.max() - img.min())
    P.normalize_and_visualize(img, normalized, title="ch0")

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Original ch0",
                                                    "Normalized ch0"]
    assert np.array_equal(_imshow_arrays(fig.axes[0])[0], img)
    assert np.array_equal(_imshow_arrays(fig.axes[1])[0], normalized)


def test_normalize_and_visualize_averages_a_multichannel_stack():
    """A 3-channel stack is shown as its channel mean, not channel 0."""
    rng = np.random.default_rng(1)
    stack = rng.random((32, 32, 3))
    P.normalize_and_visualize(stack, stack, title="")

    shown = _imshow_arrays(plt.gcf().axes[0])[0]
    assert shown.shape == (32, 32)
    assert np.allclose(shown, stack.mean(axis=-1))


def test_plot_masks():
    """Batch path: mask stack as an ndarray, cellpose-shaped ``flows``.

    The swallow here hid a real bug — an (N, H, W) mask stack was wrapped in a
    list instead of iterated, so imshow got the whole stack and raised
    "Invalid shape (1, 64, 64) for image data".
    """
    rng = np.random.default_rng(0)
    batch = rng.random((1, 64, 64, 3)).astype(np.float32)
    masks = _synth_mask()[None, ...]
    # Cellpose hands back flows as a list whose [0] is the per-image RGB flows.
    flows = [[np.zeros((64, 64, 3), dtype=np.float32)]]
    P.plot_masks(batch, masks, flows, nr=1)
    assert plt.get_fignums(), "plot_masks drew nothing"


def test_plot_masks_single_image():
    """Singleton path, as ``spacr.utils._filter_cp_masks`` actually calls it."""
    rng = np.random.default_rng(0)
    image = rng.random((64, 64, 3)).astype(np.float32)
    mask = _synth_mask()
    flow = np.zeros((64, 64, 3), dtype=np.float32)
    P.plot_masks(batch=image, masks=mask, flows=flow, nr=1)
    assert plt.get_fignums(), "plot_masks drew nothing"


# ---------------------------------------------------------------------------
# DataFrame plotters
# ---------------------------------------------------------------------------

def _screen_df(n=120):
    import pandas as pd
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "plateID": "plate1",
        "rowID": rng.choice([f"r{i}" for i in range(1, 9)], n),
        "columnID": rng.choice([f"c{i}" for i in range(1, 13)], n),
        "prc": [f"plate1_r{rng.integers(1,9)}_c{rng.integers(1,13)}"
                  for _ in range(n)],
        "recruitment": rng.normal(1.0, 0.3, n),
        "count": rng.integers(1, 100, n),
    })


def test_generate_plate_heatmap():
    """The pivot is sized by the wells present and holds the group means."""
    df = _screen_df()
    plate_map, (vmin, vmax) = P.generate_plate_heatmap(
        df, plate_number="plate1", variable="recruitment",
        grouping="mean", min_max="allq", min_count=0)

    # r1..r8 x c1..c12, read off the data rather than assumed.
    assert plate_map.shape == (8, 12)
    assert list(plate_map.index) == [f"r{i}" for i in range(1, 9)]
    assert list(plate_map.columns) == [f"c{i}" for i in range(1, 13)]
    assert vmin < vmax
    # 'allq' clips to the 2nd/98th percentile of the values plotted, so the
    # scale has to sit inside the observed range.
    observed = plate_map.to_numpy(dtype=float)
    assert vmin >= np.nanmin(observed) - 1e-9
    assert vmax <= np.nanmax(observed) + 1e-9


def test_plot_plates(tmp_path):
    """One heatmap panel per plate, on the grid the wells imply, saved once."""
    df = _screen_df()
    fig = P.plot_plates(df, variable="recruitment", grouping="mean",
                          min_max="allq", cmap="viridis", min_count=0,
                          verbose=False, dst=str(tmp_path))

    # subplots() lays out 4 columns; the 3 unused ones are deleted, and
    # seaborn adds one colorbar axes next to the single plate.
    heatmaps = [ax for ax in fig.axes if ax.get_title() == "plate1"]
    assert len(heatmaps) == 1, (
        f"expected one panel for the one plate, got titles "
        f"{[ax.get_title() for ax in fig.axes]}")
    ax = heatmaps[0]
    # The grid is read off the wells present -- r1..r8 x c1..c12 -- rather
    # than padded out to a fixed plate format. seaborn draws it as a QuadMesh,
    # whose flattened array is one cell per well.
    mesh = ax.collections[0]
    assert mesh.get_array().size == 8 * 12
    assert [t.get_text() for t in ax.get_yticklabels()] == \
        [f"r{i}" for i in range(1, 9)]
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        [f"c{i}" for i in range(1, 13)]
    _nonempty_file(tmp_path / "plate_heatmap_0.pdf")


def test_plot_histogram(tmp_path):
    """The histogram covers the data range and is written to dst."""
    df = _screen_df()
    P.plot_histogram(df, "recruitment", dst=str(tmp_path))

    ax = plt.gca()
    assert ax.get_xlabel() == "recruitment"
    assert ax.get_ylabel() == "Frequency"
    bars = [p for p in ax.patches if p.get_width() > 0]
    assert bars, "no histogram bars were drawn"
    # Every observation falls inside the bars that were drawn.
    left = min(p.get_x() for p in bars)
    right = max(p.get_x() + p.get_width() for p in bars)
    assert left <= df["recruitment"].min()
    assert right >= df["recruitment"].max()
    # Bar heights are counts, so they sum to the number of rows.
    assert sum(p.get_height() for p in bars) == len(df)
    _nonempty_file(tmp_path / "recruitment_histogram.pdf")


def test_plot_feature_importance():
    """One bar per feature, each as long as that feature's importance."""
    import pandas as pd
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(15)],
        "importance": rng.random(15),
    })
    fig = P.plot_feature_importance(df)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Feature Importance"
    bars = ax.patches
    assert len(bars) == len(df)
    # barh: the bar length is the value plotted. Compare against the frame so
    # a plot of the wrong column cannot pass.
    assert np.allclose(sorted(b.get_width() for b in bars),
                       sorted(df["importance"]))


def test_plot_permutation():
    """One bar per feature plus the std error bars."""
    import pandas as pd
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(15)],
        "importance_mean": rng.random(15),
        "importance_std": rng.random(15) * 0.1,
    })
    fig = P.plot_permutation(df)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Permutation Importance"
    assert len(ax.patches) == len(df)
    assert np.allclose(sorted(b.get_width() for b in ax.patches),
                       sorted(df["importance_mean"]))
    # xerr= produces one ErrorbarContainer; without it the std column would be
    # silently ignored and the plot would overstate the certainty.
    assert len(ax.containers) >= 1
    assert any(getattr(c, "has_xerr", False) for c in ax.containers), (
        "the importance_std column did not reach the plot as error bars")


def test_create_grouped_plot(tmp_path):
    """Three groups -> three bars, three pairwise comparisons, two files."""
    import pandas as pd
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "grp": np.repeat(["a", "b", "c"], 30),
        "val": np.concatenate([rng.normal(m, 1, 30) for m in (0.0, 3.0, 6.0)]),
    })
    fig, results = P.create_grouped_plot(
        df, grouping_column="grp", data_column="val",
        graph_type="bar", summary_func="mean",
        output_dir=str(tmp_path), save=True)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    # Bar heights are the group means, in group order.
    heights = [p.get_height() for p in ax.patches[:3]]
    assert np.allclose(heights, df.groupby("grp", observed=False)["val"].mean(),
                       atol=1e-9)
    # The stats table carries one normality test per group plus one row per
    # unordered pair for each test that ran.
    normality = results[results["Test Name"] == "Normality test"]
    assert len(normality) == 3
    pairwise = results[results["Test Name"] != "Normality test"]
    assert set(pairwise["Comparison"]) == {"a vs b", "a vs c", "b vs c"}
    # The groups are 3 sigma apart by construction, so every comparison has
    # to come out significant -- a table of NaNs would otherwise pass.
    assert (pairwise["p-value"] < 0.01).all(), results
    _nonempty_file(tmp_path / "grouped_plot.png")
    assert (tmp_path / "test_results.csv").is_file()


def test_plot_proportion_stacked_bars():
    """Each group's stack sums to 1 and covers every bin present."""
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame({
        "prc": [f"plate1_r{rng.integers(1,4)}_c{rng.integers(1,4)}"
                  for _ in range(n)],
        "group": rng.choice(["ctrl", "trt"], n),
        "bin": rng.choice([0, 1, 2], n),
    })
    chi2, pairwise, fig = P.plot_proportion_stacked_bars(
        {"verbose": False}, df, group_column="group",
        bin_column="bin", prc_column="prc", level="object")

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ctrl", "trt"]
    # 2 groups x 3 bins of stacked segments.
    assert len(ax.patches) == 2 * 3

    # Proportions, not counts: each bar stacks to exactly 1, and the segment
    # heights are the proportions actually in the data. A plot of raw counts
    # would look identical in shape and be wrong.
    for group, x in (("ctrl", 0), ("trt", 1)):
        segments = sorted(
            (p for p in ax.patches if round(p.get_x() + 0.25) == x),
            key=lambda p: p.get_y())
        heights = [p.get_height() for p in segments]
        assert sum(heights) == pytest.approx(1.0)
        expected = (df[df["group"] == group]["bin"]
                    .value_counts(normalize=True).sort_index().to_numpy())
        assert np.allclose(heights, expected)

    # The chi-squared is over the 2x3 contingency table, so 2 dof.
    assert int(chi2["degrees_of_freedom"].iloc[0]) == 2
    # One pairwise chi-squared for the single ctrl/trt pair.
    assert len(pairwise) == 1
    assert set(pairwise.iloc[0][["Group 1", "Group 2"]]) == {"ctrl", "trt"}


def test_volcano_plot(tmp_path):
    """Every gene becomes a point, at -log10(p), and the PDF is written."""
    import pandas as pd
    rng = np.random.default_rng(6)
    n = 60
    df = pd.DataFrame({
        "gene": [f"g{i}" for i in range(n)],
        "coefficient": rng.normal(0, 0.3, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-6, 1),
    })
    save = tmp_path / "volcano.pdf"
    fig, ax, hits = P.volcano_plot(df, save_path=str(save),
                                     fold_change_col="coefficient",
                                     p_value_col="p_value")

    points = np.vstack([c.get_offsets() for c in ax.collections])
    assert len(points) == n, "one point per gene"
    assert np.allclose(sorted(points[:, 0]), sorted(df["coefficient"]))
    # y is the default -log10 transform of the p-value, not the raw p.
    assert np.allclose(sorted(points[:, 1]),
                       sorted(-np.log10(df["p_value"])))
    _nonempty_file(save)


def test_create_venn_diagram(tmp_path):
    """The returned sets are the real overlap, and the PDF is written."""
    import csv
    f1 = tmp_path / "a.csv"; f2 = tmp_path / "b.csv"
    for f, genes in ((f1, [f"g{i}" for i in range(20)]),
                       (f2, [f"g{i}" for i in range(10, 30)])):
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["gene", "coefficient"])
            for g in genes:
                w.writerow([g, 0.5])
    out = P.create_venn_diagram(str(f1), str(f2), gene_column="gene",
                                  save=True, save_path=str(tmp_path / "v.pdf"))

    assert set(out["overlap"]) == {f"g{i}" for i in range(10, 20)}
    assert set(out["unique_to_file1"]) == {f"g{i}" for i in range(10)}
    assert set(out["unique_to_file2"]) == {f"g{i}" for i in range(20, 30)}
    _nonempty_file(tmp_path / "v.pdf")


def test_create_venn_diagram_filters_on_the_coefficient(tmp_path):
    """``filter_coeff`` drops genes below the threshold before intersecting.

    Without this the overlap is just "every gene in both files", and the
    coefficient column might as well not be read.
    """
    import csv
    f1 = tmp_path / "a.csv"; f2 = tmp_path / "b.csv"
    for f in (f1, f2):
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["gene", "coefficient"])
            w.writerow(["strong", 0.5])
            w.writerow(["weak", 0.01])
    out = P.create_venn_diagram(str(f1), str(f2), gene_column="gene",
                                  filter_coeff=0.1, save=False)
    assert set(out["overlap"]) == {"strong"}
    assert out["unique_to_file1"] == []
    assert out["unique_to_file2"] == []


# ---------------------------------------------------------------------------
# batch 2 — array-based plotters
# ---------------------------------------------------------------------------

def test_plot_cellpose4_output():
    """``flows`` is the *flows0* list from ``parse_cellpose4_output``.

    The old fixture handed in the raw per-image ``[rgb, dP, cellprob]`` triple,
    which is what spacr_cellpose.parse_cellpose4_output unpacks *from*, never
    what spacr.object passes on -- so imshow choked on a ragged list. The call,
    not the function, was wrong; a skip papered over the difference.
    """
    rng = np.random.default_rng(0)
    batch = rng.random((1, 32, 32, 3)).astype(np.float32)
    masks = _synth_mask(32)[None, ...]
    flows = [np.zeros((32, 32, 3), dtype=np.float32)]
    P.plot_cellpose4_output(batch, masks, flows, nr=1)
    # 3 image channels + mask + flow
    assert len(plt.gcf().axes) == 5


def test_print_mask_and_flows():
    """Image / outlined mask / flows -- three panels, and the outlines drawn."""
    stack = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    mask = _synth_mask(32)
    flows = [np.zeros((32, 32, 3), dtype=np.float32),
             np.zeros((3, 32, 32), dtype=np.float32),
             np.zeros((32, 32), dtype=np.float32)]
    P.print_mask_and_flows(stack, mask, flows, overlay=True)

    fig = plt.gcf()
    assert len(fig.axes) == 3, "flows present -> three panels"
    assert [ax.get_title() for ax in fig.axes] == [
        "Original Image", "Mask with Overlay", "Flows"]
    # Panel 0 is the first channel of the stack, greyscale.
    assert np.array_equal(_imshow_arrays(fig.axes[0])[0], stack[..., 0])
    # Panel 1 is an RGB overlay carrying pure-red contour pixels; without the
    # contours it would be a plain greyscale-to-RGB copy.
    overlay = _imshow_arrays(fig.axes[1])[0]
    assert overlay.shape == (32, 32, 3)
    red = (overlay[..., 0] == 255) & (overlay[..., 1] == 0) & \
          (overlay[..., 2] == 0)
    assert red.sum() > 0, "no mask outlines were drawn over the image"
    # ...and the outlines sit on the objects, not somewhere else.
    from scipy.ndimage import binary_dilation
    assert red[binary_dilation(mask > 0, iterations=3)].sum() == red.sum()


def test_print_mask_and_flows_without_flows_drops_the_third_panel():
    stack = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    P.print_mask_and_flows(stack, _synth_mask(32), None, overlay=False)

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Original Image", "Mask"]
    # overlay=False shows the label mask itself, not an RGB composite.
    shown = _imshow_arrays(fig.axes[1])[0]
    assert shown.shape == (32, 32)
    assert shown.max() == 6


def test_plot_resize():
    """A 2x2 grid: original/resized image on top, original/resized labels."""
    rng = np.random.default_rng(0)
    imgs = [rng.random((64, 64))]
    resized = [rng.random((32, 32))]
    labels = [_synth_mask(64)]
    rlabels = [_synth_mask(32)]
    P.plot_resize(imgs, resized, labels, rlabels)

    fig = plt.gcf()
    assert len(fig.axes) == 4
    shown = [_imshow_arrays(ax)[0] for ax in fig.axes]
    # Each panel shows its own array, at its own size -- the point of the
    # figure is the before/after comparison, so a panel showing the wrong
    # member of the pair defeats it entirely.
    assert [a.shape for a in shown] == [(64, 64), (32, 32), (64, 64), (32, 32)]
    assert np.array_equal(shown[0], imgs[0])
    assert np.array_equal(shown[1], resized[0])
    assert np.array_equal(shown[2], labels[0])
    assert np.array_equal(shown[3], rlabels[0])


def test_plot_comparison_results():
    """Each metric lands in its own panel, at its own value.

    The four panels are selected by substring (``jaccard`` / ``dice`` /
    ``boundary_f1`` / ``average_precision``) out of one melted frame, so the
    failure mode is a metric drawn on the wrong axes -- invisible unless the
    values are checked per panel.
    """
    values = {"jaccard_a_b": 0.8, "dice_a_b": 0.9,
              "boundary_f1_a_b": 0.7, "average_precision_a_b": 0.6}
    fig = P.plot_comparison_results([{"filename": "x", **values}])

    assert len(fig.axes) == 4
    assert [ax.get_ylabel() for ax in fig.axes] == [
        "Jaccard Index", "Dice Coefficient", "Boundary F1 Score",
        "Average Precision"]
    for ax, (metric, value) in zip(fig.axes, values.items()):
        assert [t.get_text() for t in ax.get_xticklabels()] == [metric]
        points = np.vstack([c.get_offsets() for c in ax.collections])
        assert len(points) == 1, f"{metric}: expected one stripplot point"
        assert points[0, 1] == pytest.approx(value)


def test_plot_comparison_results_leaves_a_missing_metric_panel_empty():
    """No dice metric supplied -> nothing drawn on the dice panel."""
    fig = P.plot_comparison_results(
        [{"filename": "x", "jaccard_a_b": 0.8}])
    dice = fig.axes[1]
    assert len(dice.collections) == 0
    assert len(dice.patches) == 0


def test_plot_lorenz_curves(tmp_path, capsys):
    """One curve per CSV plus the combined curve, each labelled with its Gini.

    No try/skip: the swallow here reported the remove_keys=None TypeError
    as a passing-looking "skipped" for as long as the bug existed.
    """
    import csv
    f = tmp_path / "counts.csv"
    with f.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["grna_name", "count"])
        for i in range(30):
            w.writerow([f"g{i}", (i + 1) * 3])
    P.plot_lorenz_curves([str(f)], name_column="grna_name",
                         value_column="count")

    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2, "one curve per plate plus the combined curve"
    labels = [ln.get_label() for ln in lines]
    assert labels[0].startswith("plate 1 (Gini:")
    assert labels[1].startswith("Combined (Gini:")
    # A Lorenz curve runs from (0, 0) to (1, 1) and never decreases.
    for line in lines:
        y = line.get_ydata()
        assert y[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(1.0)
        assert np.all(np.diff(y) >= -1e-12), "Lorenz curve must be monotone"
    # counts 3, 6, ... 90 is a linearly increasing distribution, whose Gini is
    # (n - 1) / (3n) = 29/90 ~ 0.3222. Pinning the number is what makes this a
    # test of the Gini rather than of matplotlib -- the trapezoid fix that
    # removed a 1/n bias is exactly this digit.
    gini = float(labels[0].split("Gini: ")[1].rstrip(")"))
    assert gini == pytest.approx(29 / 90, abs=1e-4)
    assert f"plate 1: Gini Coefficient = {gini:.4f}" in capsys.readouterr().out


def test_plot_lorenz_curves_saves_next_to_the_first_csv(tmp_path):
    import csv
    f = tmp_path / "counts.csv"
    with f.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["grna_name", "count"])
        for i in range(30):
            w.writerow([f"g{i}", (i + 1) * 3])
    P.plot_lorenz_curves([str(f)], name_column="grna_name",
                         value_column="count", save=True)
    _nonempty_file(tmp_path / "results" / "lorenz_curve_with_gini.pdf")


def test_plot_image_mask_overlay(tmp_path):
    """The function reads a merged ``.npy`` stack; it has no image/masks kwargs.

    The old call passed ``image=``/``masks=`` and the swallow turned the
    resulting TypeError into a green skip. The real contract (see
    spacr.core / spacr.submodules callers) is: last N planes of the stack are
    the object masks, one per non-None ``*_channel`` argument.
    """
    rng = np.random.default_rng(0)
    merged = tmp_path / "merged"
    merged.mkdir()
    stack = np.zeros((64, 64, 5), dtype=np.uint16)
    stack[..., :3] = rng.integers(0, 4000, size=(64, 64, 3))
    stack[..., 3] = _synth_mask(64)       # cell mask plane
    stack[..., 4] = _synth_mask(64)       # nucleus mask plane
    f = merged / "img_1.npy"
    np.save(f, stack)

    fig = P.plot_image_mask_overlay(
        file=str(f), channels=[0, 1, 2], cell_channel=0,
        nucleus_channel=1, pathogen_channel=None, save_pdf=True)
    assert fig is not None
    # 3 channel panels + the combined-objects panel
    assert len(fig.axes) == 4
    assert (tmp_path / "results" / "overlay" / "img_1.pdf").is_file()
